"""Aligned acoustic-history augmentation for the FireRed attention LoRA."""
from dataclasses import replace
import random

import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio

from .training_batch import FlowSample, batch_conditioning, patch_embeddings, predict, set_training
from .training_rollout import RolloutOptions, prefix_conditioning, sample_native_patches
from .training_run import isolated_rng

CONFIG = dict(schema='wav-prefix-recovery-v1', weight=.15, gain_db=[3., 6.],
              delay_ms=[2., 5.], comb_mix=.35, ramp_patches=8)
STRONGER_COMB = dict(CONFIG, comb_mix=.50)
FOUR_PATCH = dict(STRONGER_COMB, schema='wav-prefix-own-history-v1', prediction_patches=4,
                  n_timesteps=10, inference_cfg=2., stop_threshold=.5)


def read_prefix(path, samples, padded_length, stride, rate):
    """Match the clean cache's resampling grid and native left padding using only the source prefix."""
    with sf.SoundFile(path) as source:
        sr = source.samplerate
        length = (source.frames * rate + sr - 1) // sr
        left = (-length) % stride
        if source.channels != 1 or length + left != padded_length or not left < samples < padded_length:
            raise ValueError(f'Audio and clean cached patch grid disagree: {path}')
        needed = samples - left
        frames = (needed * sr + rate - 1) // rate
        waveform = torch.from_numpy(source.read(frames, dtype='float32')).unsqueeze(0)
    waveform = torchaudio.functional.resample(waveform, sr, rate)[:, :needed]
    if waveform.shape[1] != needed or not torch.isfinite(waveform).all():
        raise ValueError(f'Invalid recovery prefix: {path}')
    return F.pad(waveform, (left, 0))


def corrupt_tail(wav, ramp_samples, gain_db, delay, mix):
    """Crossfade into a rising gain and causal comb filter, keeping sample times fixed."""
    size = min(ramp_samples, wav.shape[1])
    ramp = torch.linspace(0., 1., size, device=wav.device)
    dry = wav[:, -size:]
    delayed = F.pad(wav, (delay, 0))[:, :wav.shape[1]][:, -size:]
    # Unit DC gain before the level ramp. No clipping or per-clip peak normalization.
    wet = (dry + mix * delayed) / (1. + mix)
    tail = torch.lerp(dry, wet, ramp) * torch.pow(10., gain_db * ramp / 20.)
    return torch.cat((wav[:, :-size], tail), dim=1)


@torch.no_grad()
def extend_prefixes(core, prefixes, targets, config):
    """Append up to three detached native predictions. Only target lengths are used here."""
    counts = [target.shape[1] // core.patch_size for target in targets]
    options = RolloutOptions(n_timesteps=config['n_timesteps'], inference_cfg=config['inference_cfg'])
    active = list(range(len(prefixes)))
    was_training = core.training
    try:
        core.eval()
        for hop in range(1, config['prediction_patches']):
            active = [i for i in active if counts[i] > hop]
            if not active:
                break
            cond, spk, history, stop = prefix_conditioning(core, [prefixes[i] for i in active])
            if not torch.isfinite(stop).all():
                raise FloatingPointError('Nonfinite recovery rollout STOP')
            keep = [j for j, i in enumerate(active) if prefixes[i].target.shape[1] < 6 * core.patch_size
                    or float(stop[j].sigmoid()) < config['stop_threshold']]
            active = [active[j] for j in keep]
            if not active:
                break
            generated = sample_native_patches(core, cond[keep], spk[keep], history[keep], options)
            for patch, i in zip(generated.split(1), active, strict=True):
                prefixes[i] = replace(prefixes[i], target=torch.cat((prefixes[i].target, patch), dim=1))
    finally:
        set_training(core, was_training)


def recovery_loss(tts, examples, target_rows, seed, *, config=CONFIG, patch_counts=None):
    """Supervise real future patches after damaged history, weighted equally per source window."""
    rng = random.Random(seed)
    codec, core = tts.redae, tts.tts_core
    patch = core.patch_size
    stride = patch * codec.downsample_rate
    rate = codec.sample_rate
    horizon = config.get('prediction_patches', 1)
    prefixes, targets, anchors = [], [], []
    with torch.no_grad():
        for e, row in zip(examples, target_rows, strict=True):
            count = e.target.shape[1] // patch
            if count < 4:
                continue
            anchor = rng.randint(max(3, count // 2), count - 1)
            wav = read_prefix(row['audio'], anchor * stride, count * stride, stride, rate)
            delay = max(1, round(rate * rng.uniform(*config['delay_ms']) / 1000))
            # Half the windows isolate loudness drift; the others also have spectral coloration.
            mix = config['comb_mix'] if rng.random() < .5 else 0.
            damaged = corrupt_tail(wav, stride * config['ramp_patches'], rng.uniform(*config['gain_db']), delay, mix)
            history = codec.encode(damaged.to(e.target.device), rate).float()
            if history.shape != e.target[:, :anchor * patch].shape or not torch.isfinite(history).all():
                raise ValueError(f'Invalid re-encoded recovery history: {row["audio"]}')
            prefixes.append(replace(e, target=history, target_embeds=None))
            anchors.append(anchor)
            targets.append(e.target[:, anchor * patch:(anchor + horizon) * patch].detach())
    if not targets:
        return examples[0].target.new_zeros(()), 0
    if horizon == 1:
        cond, speaker, history, _ = prefix_conditioning(core, prefixes)
        clean = torch.cat(targets)
        lengths = [1] * len(targets)
    else:
        # Keep the rollout's initial ODE noise independent of the flow-matching
        # regression noise below, including on the first patch of a microbatch.
        with isolated_rng(seed + 1_000_000):
            extend_prefixes(core, prefixes, targets, config)
        # A single causal backbone pass per utterance can score every rollout position.
        # The state at index j sees fed patches up to j-1. The decoder history
        # is gathered with the same boundary, including when j is the final state.
        samples, histories, future = [], [], []
        lengths = []
        for e, start, target in zip(prefixes, anchors, targets, strict=True):
            indices = torch.arange(start, e.target.shape[1] // patch + 1, device=target.device)
            samples.append(FlowSample(indices, None, None, None))
            fed = F.pad(torch.cat([e.prompt, e.target], 1), (0, 0, core.history_length, 0))
            offsets = e.prompt.shape[1] + indices[:, None] * patch
            histories.append(fed[0, offsets + torch.arange(core.history_length, device=target.device)])
            future.append(target[:, :len(indices)*patch].reshape(-1, patch, target.shape[-1]))
            lengths.append(len(indices))
        cond, speaker, *_ = batch_conditioning(core, prefixes, samples, patch_embeddings(core, prefixes), 'native')
        history, clean = torch.cat(histories), torch.cat(future)
    generator = torch.Generator(device=clean.device).manual_seed(seed)
    noise = torch.randn(clean.shape, device=clean.device, dtype=clean.dtype, generator=generator)
    times = 1 - (torch.randn(len(clean), device=clean.device, generator=generator) - .4).sigmoid()
    drop = torch.zeros(len(clean), dtype=torch.bool, device=clean.device)
    velocity = predict(core, cond, speaker, (clean, history, noise, times, drop))
    errors = (velocity - (clean - noise).float()).square().mean(dim=(1, 2))
    if patch_counts is not None:
        for i in range(horizon):
            patch_counts[i] += sum(length > i for length in lengths)
    per_window = errors if horizon == 1 else torch.stack([group.mean() for group in errors.split(lengths)])
    return per_window.sum() / len(examples), len(targets)

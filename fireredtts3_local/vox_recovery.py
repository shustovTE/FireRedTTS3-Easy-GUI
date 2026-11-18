"""Aligned acoustic-history augmentation for the FireRed attention LoRA."""
from dataclasses import replace
import random

import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio

from .training_batch import predict
from .training_rollout import prefix_conditioning

CONFIG = dict(schema='wav-prefix-recovery-v1', weight=.15, gain_db=[3., 6.],
              delay_ms=[2., 5.], comb_mix=.35, ramp_patches=8)
STRONGER_COMB = dict(CONFIG, comb_mix=.50)


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


def recovery_loss(tts, examples, target_rows, seed, *, config=CONFIG):
    """Predict one clean next patch for each eligible utterance after a corrupted late prefix."""
    rng = random.Random(seed)
    codec, core = tts.redae, tts.tts_core
    patch = core.patch_size
    stride = patch * codec.downsample_rate
    rate = codec.sample_rate
    prefixes, targets = [], []
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
            targets.append(e.target[:, anchor * patch:(anchor + 1) * patch].detach())
    if not targets:
        return examples[0].target.new_zeros(()), 0
    cond, speaker, history, _ = prefix_conditioning(core, prefixes)
    clean = torch.cat(targets)
    generator = torch.Generator(device=clean.device).manual_seed(seed)
    noise = torch.randn(clean.shape, device=clean.device, dtype=clean.dtype, generator=generator)
    times = 1 - (torch.randn(len(clean), device=clean.device, generator=generator) - .4).sigmoid()
    drop = torch.zeros(len(clean), dtype=torch.bool, device=clean.device)
    velocity = predict(core, cond, speaker, (clean, history, noise, times, drop))
    per_window = (velocity - (clean - noise).float()).square().mean(dim=(1, 2))
    return per_window.sum() / len(examples), len(targets)

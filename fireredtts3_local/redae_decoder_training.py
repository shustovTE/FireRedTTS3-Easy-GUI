"""Waveform supervision for a FireRed RedAE decoder attention adapter."""
import json
from pathlib import Path
import random

import torch
import torch.nn.functional as F

from .training_data import file_digest


FOLDER = 'redae_decoder'
CONFIG = dict(version=1, profile='decoder-qkvo', rank=8, alpha=8, lr=1e-5,
              fft_sizes=[512, 1024, 2048], tail_seconds=1.28,
              supervision='aligned-clean-wav-half-damaged-v1')


def saved_hashes(checkpoint):
    return {f'{FOLDER}/{name}': file_digest(checkpoint/FOLDER/name)
            for name in ('adapter_config.json', 'adapter_model.safetensors')}


def verify(checkpoint):
    record = json.loads((checkpoint/'checkpoint.json').read_text())
    if record['recipe'].get('redae_decoder') != CONFIG or record.get('decoder_hashes') != saved_hashes(checkpoint):
        raise ValueError(f'Unexpected or incomplete RedAE decoder checkpoint: {checkpoint}')


def has_decoder(checkpoint):
    if not checkpoint:
        return False
    path = Path(checkpoint)
    if (path/FOLDER).exists():
        return True
    # Detect a missing decoder even if only the backbone adapter was copied.
    for filename in ('checkpoint.json', 'firered_lora_metadata.json'):
        meta = path/filename
        if meta.is_file() and json.loads(meta.read_text()).get('recipe', {}).get('redae_decoder'):
            return True
    return False


def configure(codec, checkpoint=None, *, training):
    from peft import LoraConfig, PeftModel, get_peft_model
    codec.eval().requires_grad_(False)
    decoder = codec.decoder
    base = decoder.qwen3
    targets = [f'layers.{i}.self_attn.{proj}_proj' for i in range(len(base.layers)) for proj in 'qkvo']
    if checkpoint is None:
        config = LoraConfig(r=CONFIG['rank'], lora_alpha=CONFIG['alpha'], target_modules=targets,
                            lora_dropout=0., bias='none')
        adapter = get_peft_model(base, config)
    else:
        verify(checkpoint)
        adapter = PeftModel.from_pretrained(base, str(checkpoint/FOLDER), is_trainable=training)
    if set(adapter.peft_config['default'].target_modules) != set(targets) or not targets:
        raise ValueError('RedAE adapter must cover Q/K/V/O in each decoder layer')
    decoder.qwen3 = adapter
    if training:
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    else:
        decoder.requires_grad_(False)
    decoder.train(training)
    return adapter


def spectral_loss(pred, target):
    """Compare both magnitude and log magnitude at several resolutions in native waveform units."""
    loss = F.l1_loss(pred, target)
    for size in CONFIG['fft_sizes']:
        window = torch.hann_window(size, device=pred.device)
        def magnitude(wav):
            return torch.stft(wav, size, hop_length=size//4, window=window, pad_mode='constant',
                              return_complex=True).abs()
        p, t = magnitude(pred), magnitude(target)
        convergence = torch.linalg.vector_norm(p-t) / torch.linalg.vector_norm(t).clamp_min(1e-2)
        log_error = F.l1_loss(torch.log(p + .01), torch.log(t + .01))
        loss = loss + (convergence + log_error) / len(CONFIG['fft_sizes'])
    return loss


def waveform_loss(tts, loaded_row, *, damaged, seed):
    from .history_probe import read_audio
    from .vox_recovery import FOUR_PATCH, corrupt_tail
    _, cached, _, target, _ = loaded_row
    codec = tts.redae
    # The cache uses the frozen encoder and left-pads to a whole FireRed patch.
    with torch.no_grad():
        clean, padded, left = read_audio(tts, target)
        latents = cached['latents'].to(tts.device)
        if latents.shape[1] * codec.downsample_rate != padded.shape[-1]:
            raise ValueError(f'Cached latent grid and waveform disagree: {target["audio"]}')
        if damaged:
            rng = random.Random(seed)
            delay = max(1, round(codec.sample_rate * rng.uniform(*FOUR_PATCH['delay_ms']) / 1000))
            mix = FOUR_PATCH['comb_mix'] if rng.random() < .5 else 0.
            ramp = FOUR_PATCH['ramp_patches'] * tts.tts_core.patch_size * codec.downsample_rate
            distorted = corrupt_tail(padded, ramp, rng.uniform(*FOUR_PATCH['gain_db']), delay, mix)
            latents = codec.encode(distorted, codec.sample_rate)
    # RedAE.decode is decorated with no_grad. Use the same native decoder forward with gradients.
    pred = codec.decoder(latents.to(codec.decoder.in_proj.weight.dtype)).float()[:, left:]
    if pred.shape != clean.shape:
        raise ValueError('RedAE reconstruction is not sample-aligned with the target WAV')
    tail = round(CONFIG['tail_seconds'] * codec.sample_rate)
    loss = (spectral_loss(pred, clean) + spectral_loss(pred[:, -tail:], clean[:, -tail:])) / 2
    if not torch.isfinite(loss):
        raise FloatingPointError(f'Nonfinite decoder reconstruction loss: {target["audio"]}')
    return loss

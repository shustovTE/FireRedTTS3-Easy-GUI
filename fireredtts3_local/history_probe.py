"""Inference-only history intervention; the free control uses upstream generate."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio

from .native_comparison import signal_levels
from .training_data import file_digest, plain_text
from .training_run import isolated_rng


def levels(samples, rate):
    result = signal_levels(samples, rate)
    # Non-overlapping windows also work on the short Dialogs control.
    size = max(1, min(10*rate, len(samples)//3))
    rms = lambda a: 20*np.log10(max(float(np.sqrt(np.mean(np.asarray(a, dtype=np.float64)**2))), 1e-12))
    result.update(rms_window_seconds=size/rate,
                  end_minus_start_rms_db=float(rms(samples[-size:])-rms(samples[:size])))
    return result


def reconstruction_metrics(source, reconstructed, rate):
    """Aligned codec measurements; never apply these as free-generation scores."""
    if source.shape != reconstructed.shape:
        raise ValueError('Codec comparison requires exactly aligned sample counts')
    a, b = np.asarray(source, dtype=np.float64), np.asarray(reconstructed, dtype=np.float64)
    gains = []
    for start in range(0, len(a), rate):
        x, y = a[start:start+rate], b[start:start+rate]
        power = np.mean(x*x)
        if power >= 1e-5:  # -50 dBFS; skip silent source windows
            gains.append(dict(seconds=start/rate, db=float(10*np.log10(max(np.mean(y*y), 1e-12)/power))))
    first = [v['db'] for v in gains if v['seconds'] < len(a)/rate/3]
    last = [v['db'] for v in gains if v['seconds'] >= 2*len(a)/rate/3]
    nfft, hop = 1024, 256
    def spectrum(wave):
        wave = np.pad(wave, (0, max(0, nfft-len(wave))))
        frames = np.lib.stride_tricks.sliding_window_view(wave, nfft)[::hop]
        return np.abs(np.fft.rfft(frames*np.hanning(nfft), axis=1))
    x, y = spectrum(a), spectrum(b)
    return dict(active_window_gain_db=gains,
        tail_gain_minus_head_gain_db=float(np.median(last)-np.median(first)) if first and last else None,
        snr_db=float(10*np.log10(max(np.sum(a*a), 1e-12)/max(np.sum((a-b)**2), 1e-12))),
        spectral_convergence=float(np.linalg.norm(y-x)/max(np.linalg.norm(x), 1e-12)),
        note='Time-aligned codec reconstruction only. Source dynamics are retained; no loudness correction or perceptual threshold.')


def read_audio(tts, row):
    if file_digest(row['audio']) != row['audio_sha256']:
        raise ValueError('Diagnostic source audio changed')
    audio, rate = torchaudio.load(row['audio'])
    if audio.ndim != 2 or audio.shape[0] != 1 or not torch.isfinite(audio).all():
        raise ValueError('Require finite mono source audio')
    rate_out = tts.redae.sample_rate
    audio = torchaudio.functional.resample(audio, rate, rate_out)
    padded = tts.redae.pad_to_multiple_of(audio, tts.redae.downsample_rate*tts.tts_core.patch_size)
    return audio.to(tts.device), padded.to(tts.device), padded.shape[1]-audio.shape[1]


@torch.inference_mode()
def prepare_inputs(tts, case):
    source, target_audio, target_pad = read_audio(tts, case['target'])
    _, prompt_audio, prompt_pad = read_audio(tts, case['reference'])
    prompt = tts.redae.encode(prompt_audio, tts.redae.sample_rate).float()
    target = tts.redae.encode(target_audio, tts.redae.sample_rate).float()
    speaker = tts.spk_extractor(prompt_audio, tts.redae.sample_rate).to(tts.device)
    text, language, sentences = tts._apply_frontend(
        text=plain_text(case['target']['text']), language=case['target']['language'], do_split=False)
    if len(sentences) != 1:
        raise ValueError('History diagnostics must use an unsplit complete transcript')
    tokens = tts._tokenize_text(f'<|{language}|><|sot|>{plain_text(case["reference"]["text"])}{text}<|eot|>')
    return dict(source=source, prompt=prompt, target=target, speaker=speaker, tokens=tokens,
                target_pad=target_pad, prompt_pad=prompt_pad, prompt_samples=prompt_audio.shape[1],
                source_samples=source.shape[1], rate=tts.redae.sample_rate, effective_text=text)


def cpu_audio(wave):
    if wave.ndim != 2 or wave.shape[0] != 1 or not torch.isfinite(wave).all():
        raise ValueError('Expected finite mono diagnostic waveform')
    return wave[0].detach().float().cpu().numpy()


@torch.inference_mode()
def decode_target(tts, inputs, latents, *, fixed):
    # Native decoding includes the real reference; never decode predicted
    # patches separately or assemble sentence/patch waveforms.
    wave, rate = tts.redae.decode(torch.cat([inputs['prompt'], latents], 1))
    if rate != inputs['rate']:
        raise ValueError('Unexpected codec sample rate')
    start = inputs['prompt_samples'] + (inputs['target_pad'] if fixed else 0)
    end = start+inputs['source_samples'] if fixed else wave.shape[1]
    if wave.shape[1] < end:
        raise ValueError('Codec returned fewer samples than the padded latent sequence')
    return cpu_audio(wave[:, start:end])


@torch.inference_mode()
def codec_check(tts, inputs):
    wave, rate = tts.redae.decode(inputs['target'])
    start = inputs['target_pad']
    if rate != inputs['rate'] or wave.shape[1] < start+inputs['source_samples']:
        raise ValueError('Unexpected standalone codec output length/rate')
    return dict(source=cpu_audio(inputs['source']),
                codec=cpu_audio(wave[:, start:start+inputs['source_samples']]),
                codec_with_reference=decode_target(tts, inputs, inputs['target'], fixed=True))


@torch.inference_mode()
def fixed_history(core, inputs, *, teacher, seed, settings):
    """Cached native primitives, with exactly N patches; STOP is observed only.

    Both histories are replaced together: backbone patch inputs and DiT history.
    Only target[:j] may condition prediction j. The output patch is always a
    fresh ODE sample, never copied from the target, including in teacher mode.
    """
    prompt, target = inputs['prompt'], inputs['target']
    if target.shape[1] % core.patch_size or prompt.shape[1] % core.patch_size:
        raise ValueError('Complete latent patches are required')
    count = target.shape[1]//core.patch_size
    with isolated_rng(seed):
        text_embeds = core.backbone_llm.embed_tokens(inputs['tokens'])
        prompt_embeds = core.patch_encoder(prompt)
        embeddings = torch.cat([core.spk_proj_llm(inputs['speaker']).unsqueeze(1), text_embeds, prompt_embeds], 1)
        history = F.pad(prompt, (0, 0, core.history_length, 0))
        cond = embeddings.new_zeros(1, core.history_patches, embeddings.shape[-1])
        spk = core.spk_proj_dit(inputs['speaker'])
        times = torch.linspace(0, 1, settings['n_timesteps']+1, device=target.device)
        times = 1-torch.cos(times*.5*torch.pi)
        cache, generated, stops = None, [], []
        for index in range(count+1):
            hidden, cache = core._backbone_one_step(embeddings, cache=cache)
            stops.append(float(core.stop_head(hidden[:, -1]).squeeze(-1).sigmoid().item()))
            if index == count:
                break
            current = hidden[:, -prompt_embeds.shape[1]:] if index == 0 else hidden[:, -1:]
            cond = torch.cat([cond, current], 1)
            prediction = core._flow_one_step(history[:, -core.history_length:],
                core.dit_head(cond[:, -(core.history_patches+1):]), spk, times, settings['inference_cfg'])
            generated.append(prediction)
            fed = target[:, index*core.patch_size:(index+1)*core.patch_size] if teacher else prediction
            embeddings = core.patch_encoder(fed)
            history = torch.cat([history, fed], 1)
        return torch.cat(generated, 1), stops


@torch.inference_mode()
def native_free(core, inputs, seed, settings):
    stops = []
    hook = core.stop_head.register_forward_hook(lambda _m, _a, out: stops.append(float(out.sigmoid().item())))
    try:
        with isolated_rng(seed):
            result = core.generate(spk_emb=inputs['speaker'], text_tokens=inputs['tokens'],
                prompt_latents=inputs['prompt'], n_timesteps=settings['n_timesteps'],
                inference_cfg=settings['inference_cfg'], stop_threshold=settings['stop_threshold'],
                min_gen_steps=6, max_gen_steps=400)
        return result[:, inputs['prompt'].shape[1]:], stops
    finally:
        hook.remove()


def close_check(actual, expected):
    if actual.shape != expected.shape or not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
        return dict(passed=False, reason='shape or finite-value mismatch')
    diff = (actual-expected).float()
    absolute = float(diff.abs().max())
    relative = float(diff.norm()/expected.float().norm().clamp_min(1e-12))
    return dict(passed=absolute <= 2e-5 or relative <= .02, max_abs=absolute, relative_l2=relative)


@torch.inference_mode()
def history_check(tts, inputs, seed, settings):
    core = tts.tts_core
    free, free_stops = native_free(core, inputs, seed, settings)
    own, own_stops = fixed_history(core, inputs, teacher=False, seed=seed, settings=settings)
    teacher, teacher_stops = fixed_history(core, inputs, teacher=True, seed=seed, settings=settings)
    common = min(free.shape[1], own.shape[1])
    checks = dict(native_common_prefix=close_check(own[:, :common], free[:, :common]),
                  first_patch=close_check(teacher[:, :core.patch_size], own[:, :core.patch_size]),
                  common_patches=common//core.patch_size)
    # A discrepancy needs investigation; it must not masquerade as a history effect.
    if not checks['native_common_prefix']['passed'] or not checks['first_patch']['passed']:
        raise ValueError(f'Native/history control mismatch: {checks}')
    result = []
    for mode, latent, stops in [('free', free, free_stops), ('self_fixed', own, own_stops), ('teacher', teacher, teacher_stops)]:
        patches = latent.reshape(-1, core.patch_size, latent.shape[-1]).float()
        baseline = inputs['target'].reshape(-1, core.patch_size, latent.shape[-1]).float()
        patch_rows = [dict(index=i, rms=float(p.square().mean().sqrt()), peak=float(p.abs().max()),
            target_rms=float(baseline[i].square().mean().sqrt()) if i<len(baseline) else None)
            for i,p in enumerate(patches)]
        first_stop = next((i for i,p in enumerate(stops) if i>=6 and p>=settings['stop_threshold']), None)
        result.append(dict(mode=mode, samples=decode_target(tts, inputs, latent, fixed=mode!='free'),
            generated_patches=len(patches), target_patches=len(baseline), stop_scores=stops,
            first_eligible_stop_patch=first_stop,
            finish_reason=('stop' if first_stop is not None else 'max_patches') if mode=='free' else 'fixed_target_length',
            latent_by_patch=patch_rows, controls=checks))
    return result

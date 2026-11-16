"""Free-running regression checks during training; never a perceptual quality certificate."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
import soundfile as sf
import torch
import torchaudio

from .native_comparison import signal_levels
from .training_data import file_digest, language_name, plain_text

SETTINGS = dict(stop_threshold=.5, n_timesteps=10, inference_cfg=2., do_split=False)


def load_benchmark(path):
    path = Path(path).resolve()
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Generation benchmark requires unique nonempty cases')
    result = []
    for row in rows:
        ref = Path(row['reference_audio'])
        ref = ref.resolve() if ref.is_absolute() else (path.parent / ref).resolve()
        if not ref.is_file() or not row['text'].strip() or not row['reference_text'].strip():
            raise ValueError('Generation benchmark requires audio and exact reference/target text')
        info = sf.info(ref)
        if info.channels != 1 or not 0 < info.duration <= 30:
            raise ValueError('Generation references must be nonempty mono <=30s')
        result.append({**row, 'reference_audio': str(ref), 'reference_sha256': file_digest(ref),
                       'language': language_name(row.get('language', 'Russian')),
                       'seed': int(row.get('seed', 1234))})
    digest = hashlib.sha256(json.dumps(dict(cases=result, settings=SETTINGS), sort_keys=True).encode()).hexdigest()
    return result, digest


def measure(samples, rate, scores):
    levels = signal_levels(samples, rate)
    n = min(10 * rate, len(samples))
    def rms(values):
        return float(20 * np.log10(max(float(np.sqrt(np.mean(np.asarray(values, dtype=np.float64)**2))), 1e-12)))
    levels['end_minus_start_rms_db'] = rms(samples[-n:]) - rms(samples[:n])
    if not scores:
        raise RuntimeError('Generation monitor did not observe the stop head')
    stopped = len(scores) > 6 and scores[-1] >= SETTINGS['stop_threshold']
    levels.update(finish_reason='stop' if stopped else 'max_patches',
                  generated_patches=len(scores) - int(stopped))
    return levels


def failures(current, baseline=None):
    reasons = []
    if current['finish_reason'] != 'stop':
        reasons.append('generation_limit')
    if current['fraction_at_or_above_full_scale'] > .001:
        reasons.append('raw_clipping_above_0.1_percent')
    if baseline and current['end_minus_start_rms_db'] > baseline['end_minus_start_rms_db'] + 3.:
        reasons.append('level_drift_above_baseline_plus_3db')
    return reasons


def monitor(tts, benchmark, output, step, expected_fingerprint, offloaded=False, log=print):
    from .training_run import isolated_rng
    from .training_batch import set_training
    rows, digest = load_benchmark(benchmark)
    if digest != expected_fingerprint:
        raise ValueError('Generation benchmark/reference changed since run preparation')
    output = Path(output)
    baseline_path = output / 'generation_baseline.json'
    baseline = json.loads(baseline_path.read_text()) if step else None
    if baseline and baseline['fingerprint'] != digest:
        raise ValueError('Generation baseline fingerprint mismatch')
    before = {r['id']: r for r in baseline['results']} if baseline else {}
    root = output / 'generation'
    root.mkdir(exist_ok=True)
    folder = Path(tempfile.mkdtemp(prefix=f'step-{step:06d}-', dir=root))
    report = dict(step=step, fingerprint=digest, settings=SETTINGS, complete=False, results=[])
    core = tts.tts_core
    was_training = core.training
    try:
        core.eval()
        if offloaded:
            tts.redae.to(tts.device)
            tts.spk_extractor.to(tts.device)
        for i, row in enumerate(rows):
            reference, sr = torchaudio.load(row['reference_audio'])
            scores, latents = [], []
            hook = core.stop_head.register_forward_hook(lambda _m, _a, value:
                scores.append(float(value.detach().float().sigmoid().flatten()[-1])))
            def observe(_module, args):
                value = args[0].detach().float()
                latents.append(dict(frames=value.shape[1], rms=float(value.square().mean().sqrt()),
                                    peak=float(value.abs().max())))
            latent_hook = core.patch_encoder.register_forward_pre_hook(observe)
            try:
                with isolated_rng(row['seed']), torch.inference_mode():
                    wav, rate = tts.generate(language=row['language'], prompt_text=plain_text(row['reference_text']),
                        prompt_audio=reference, prompt_audio_sr=sr, text=plain_text(row['text']),
                        seed=row['seed'], **SETTINGS)
            finally:
                hook.remove()
                latent_hook.remove()
            samples = wav.detach().float().cpu().flatten().numpy()
            levels = measure(samples, rate, scores)
            audio = f'case-{i:05d}'
            sf.write(folder / f'{audio}.float.wav', samples, rate, subtype='FLOAT')
            gain = min(1., .98 / max(levels['peak'], 1e-12))
            sf.write(folder / f'{audio}.wav', samples * gain, rate, subtype='PCM_16')
            result = dict(id=row['id'], seed=row['seed'], **levels, pcm_gain=gain,
                          float_audio=f'{audio}.float.wav', audio=f'{audio}.wav',
                          stop_scores=scores, latent_levels=latents)
            result['failures'] = failures(result, before.get(row['id']))
            report['results'].append(result)
            log(f"[generation] step={step} id={row['id']} seconds={levels['seconds']:.2f} "
                f"finish={levels['finish_reason']} failures={result['failures']}")
        report['complete'] = True
        report['passed'] = all(not r['failures'] for r in report['results'])
        (folder / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
        if not step:
            baseline_path.write_text(json.dumps(report, indent=2, allow_nan=False))
        summary = dict(step=step, passed=report['passed'], report=str((folder / 'report.json').resolve()),
                       failures={r['id']: r['failures'] for r in report['results'] if r['failures']},
                       note='Engineering regression checks only; listening, ASR and voice similarity are still required.')
        with (output / 'generation_metrics.jsonl').open('a') as stream:
            stream.write(json.dumps(summary) + '\n')
        return summary
    finally:
        if offloaded:
            tts.redae.to('cpu')
            tts.spk_extractor.to('cpu')
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        set_training(core, was_training)

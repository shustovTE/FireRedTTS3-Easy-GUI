"""Prepare a levelled copy of the audited Dialogs and Spontaneous WAVs."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

from filelock import FileLock
import numpy as np
import soundfile as sf

from . import voxcpm_data
from .training_data import file_digest
from .workspace_paths import local_path

OUTPUT = 'training/datasets/firered-vox-cap30m-levels'
REPORT = 'level_report.json'
METHOD = dict(version='active-rms-envelope-r128-v1', frame_ms=50, window_s=3., max_cut_db=6.,
              max_db_per_second=2., deadband_db=1., anchor_percentile=25,
              absolute_gate_dbfs=-60., relative_gate_db=25., target_lufs=-23.,
              tolerance_lu=.2, true_peak_ceiling_dbtp=-1., limiter_margin_db=.3)


def level_audio(samples, rate):
    """Attenuate sustained louder passages, with a slow, sample-continuous gain."""
    x = np.asarray(samples, dtype=np.float64)
    if x.ndim != 1 or not len(x) or rate <= 0 or not np.isfinite(x).all():
        raise ValueError('Expected nonempty finite mono audio')
    hop = max(1, round(rate * METHOD['frame_ms'] / 1000))
    starts = np.arange(0, len(x), hop)
    lengths = np.diff(np.append(starts, len(x)))
    centers = starts + (lengths - 1) / 2

    def frame_db(signal):
        energy = np.add.reduceat(signal * signal, starts) / lengths
        return 10 * np.log10(np.maximum(energy, 1e-14))

    before = frame_db(x)
    gate = max(METHOD['absolute_gate_dbfs'], np.percentile(before, 90) - METHOD['relative_gate_db'])
    active = np.flatnonzero(before > gate)
    cut = np.zeros(len(starts))
    if len(active) >= 2:
        positions = centers[active]
        radius = rate * METHOD['window_s'] / 2
        left = np.searchsorted(positions, positions - radius)
        right = np.searchsorted(positions, positions + radius, side='right')
        levels = before[active]
        envelope = np.array([np.median(levels[a:b]) for a, b in zip(left, right)])
        anchor = np.percentile(envelope, METHOD['anchor_percentile'])
        desired = np.clip(envelope - anchor - METHOD['deadband_db'], 0, METHOD['max_cut_db'])
        cut = np.interp(centers, positions, desired)
        # Bound both attack and release in dB/second, including across pauses.
        slopes = np.diff(centers) * METHOD['max_db_per_second'] / rate
        for i, delta in enumerate(slopes, 1):
            cut[i] = min(cut[i], cut[i-1] + delta)
        for i in range(len(slopes)-1, -1, -1):
            cut[i] = min(cut[i], cut[i+1] + slopes[i])

    y = x * 10 ** (-np.interp(np.arange(len(x)), centers, cut) / 20)

    def drift(signal):
        # Compare medians in disjoint first/last thirds of active frames.
        n = len(active) // 3
        db = frame_db(signal)
        return float(np.median(db[active[-n:]]) - np.median(db[active[:n]])) if n else None

    stats = dict(active_frames=len(active), frames=len(x), samplerate=rate,
                 max_envelope_cut_db=float(cut.max()), active_drift_before_db=drift(x))
    return y.astype(np.float32), stats, drift


def ffmpeg_audio(samples, rate, audio_filter, *, meter=False):
    args = ['ffmpeg', '-nostdin', '-hide_banner', '-nostats', '-loglevel', 'info' if meter else 'error',
            '-f', 'f32le', '-ar', str(rate), '-ac', '1', '-i', 'pipe:0', '-filter_threads', '1',
            '-af', audio_filter, '-f', 'null' if meter else 'f32le', 'pipe:1']
    proc = subprocess.run(args, input=np.asarray(samples, dtype='<f4').tobytes(), capture_output=True)
    log = proc.stderr.decode(errors='replace')
    if proc.returncode:
        raise RuntimeError(f'FFmpeg audio filter failed: {log[-1500:]}')
    if meter:
        data = json.loads(log[log.rindex('{'):log.rindex('}')+1])
        return float(data['input_i']), float(data['input_tp'])
    return np.frombuffer(proc.stdout, dtype='<f4')


def measure_loudness(samples, rate):
    # Read the input measurements from the R128 meter, discarding its output.
    gate_samples = (rate * 4 + 9) // 10
    if len(samples) < gate_samples:
        samples = np.pad(samples, (0, gate_samples - len(samples)))
    return ffmpeg_audio(samples, rate, 'loudnorm=print_format=json', meter=True)


def normalize_recording(path, dest):
    samples, rate = sf.read(path, dtype='float32')
    y, stats, drift = level_audio(samples, rate)
    target, ceiling = METHOD['target_lufs'], METHOD['true_peak_ceiling_dbtp']
    loudness, peak = measure_loudness(y, rate)
    stats['after_envelope_lufs'] = loudness if np.isfinite(loudness) else None
    gain_db = 0.
    limited = False
    if not np.isfinite(loudness):
        if not np.max(np.abs(y)):
            raise ValueError(f'Silent audio has no measurable loudness: {path}')
        # Bring very quiet but nonzero speech above the absolute R128 gate.
        gain_db = -12. - 20 * np.log10(float(np.max(np.abs(y))))
        y = y * 10 ** (gain_db / 20)
        loudness, peak = measure_loudness(y, rate)
    if not np.isfinite([loudness, peak]).all():
        raise ValueError(f'No measurable programme loudness: {path}')

    for _ in range(5):
        correction = target - loudness
        gain_db += correction
        y = y * 10 ** (correction / 20)
        if peak + correction > ceiling - .05:
            # Rare high-crest recordings. Oversample, limit only peaks, compensate
            # lookahead latency, then restore the original sample rate and length.
            limit = 10 ** ((ceiling - METHOD['limiter_margin_db']) / 20)
            filters = (f'apad=pad_dur=0.1,aresample={rate*4},'
                       f'alimiter=limit={limit}:attack=5:release=50:level=0:latency=1,'
                       f'aresample={rate},atrim=end_sample={len(samples)}')
            y = ffmpeg_audio(y, rate, filters)
            limited = True
        if len(y) != len(samples) or not np.isfinite(y).all():
            raise ValueError(f'Invalid levelled waveform: {path}')
        # Keep headroom before conversion to integer PCM, even for atypical inputs.
        if np.max(np.abs(y)) >= 1:
            raise ValueError(f'Peak limiter failed to leave sample headroom: {path}')
        sf.write(dest, y, rate, subtype='PCM_24')
        y, _ = sf.read(dest, dtype='float32')
        loudness, peak = measure_loudness(y, rate)
        if abs(loudness - target) <= METHOD['tolerance_lu'] and peak <= ceiling:
            return dict(**stats, programme_gain_db=float(gain_db), peak_limiter_used=limited,
                        output_lufs=loudness, output_true_peak_dbtp=peak, active_drift_after_db=drift(y))
    raise ValueError(f'Cannot meet loudness/true-peak target for {path}: {loudness} LUFS, {peak} dBTP')


def read_report(folder):
    report = json.loads((folder / REPORT).read_text(encoding='utf-8'))
    source = local_path(report['source']).resolve()
    if (report['method'] != METHOD or file_digest(source / 'report.json') != report['source_sha256']
            or voxcpm_data._digest_json(report['files']) != report['files_sha256']):
        raise ValueError('Levelled audio method, source report or index changed')
    return source, report


def verify_files(folder, report, expected):
    if set(expected) != set(report['files']):
        raise ValueError('Levelled audio index differs from the original train/validation inventory')
    for sha, record in report['files'].items():
        path = folder / 'audio' / f'{sha}.wav'
        if file_digest(path) != record['audio_sha256']:
            raise ValueError(f'Levelled WAV changed: {path}')


def summarize(files, rows):
    measured = [files[r['audio_sha256']] for r in rows]
    summary = dict(clips=len(rows), envelope_adjusted=sum(r['max_envelope_cut_db'] > 0 for r in measured),
                   peak_limited=sum(r['peak_limiter_used'] for r in measured),
                   min_lufs=min(r['output_lufs'] for r in measured), max_lufs=max(r['output_lufs'] for r in measured),
                   max_true_peak_dbtp=max(r['output_true_peak_dbtp'] for r in measured))
    for when in ('before', 'after'):
        values = [r[f'active_drift_{when}_db'] for r in measured if r[f'active_drift_{when}_db'] is not None]
        summary[when] = dict(measured=len(values), rise_over_3db=sum(v > 3 for v in values),
                             median_abs_drift_db=float(np.median(np.abs(values))) if values else None)
    return summary


def prepare(source, output):
    source, output = local_path(source).resolve(), output.resolve()
    if output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError('Use separate source and destination directories')
    print(f'[levels] Auditing original audio and pairs in {source}', flush=True)
    audit = voxcpm_data.audit(source)
    if audit['schema'] != voxcpm_data.FAIR_SCHEMA:
        raise ValueError('Expected the audited Dialogs and Spontaneous preparation')
    splits = {split: voxcpm_data._read_rows(source / f'{split}-provenance.jsonl')
              for split in ('train', 'validation')}
    audio = {r['audio_sha256']: local_path(r['audio']) for rows in splits.values() for r in rows}
    output.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output) + '.lock'), tempfile.TemporaryDirectory(dir=output.parent) as tmp:
        if output.exists():
            existing_source, report = read_report(output)
            if existing_source != source:
                raise ValueError('Existing levelled audio belongs to a different source')
            verify_files(output, report, audio)
            return output / REPORT
        if not shutil.which('ffmpeg'):
            raise RuntimeError('Preparing levelled WAVs requires the ffmpeg executable on PATH')
        stage = Path(tmp) / 'dataset'
        (stage / 'audio').mkdir(parents=True)
        def render(sha):
            path = audio[sha]
            if file_digest(path) != sha:
                raise ValueError(f'Source WAV changed since audit: {path}')
            dest = stage / 'audio' / f'{sha}.wav'
            stats = normalize_recording(path, dest)
            return sha, dict(**stats, audio_sha256=file_digest(dest), pcm_sha256=voxcpm_data._pcm_digest(dest))

        files = {}
        hashes = sorted(audio)
        with ThreadPoolExecutor(max_workers=3) as pool:
            for start in range(0, len(hashes), 30):
                files.update(pool.map(render, hashes[start:start+30]))
                print(f'[levels] {len(files)}/{len(audio)} WAV', flush=True)
        summary = {split: summarize(files, rows) for split, rows in splits.items()}
        report = dict(source=str(source), source_sha256=file_digest(source / 'report.json'), method=METHOD,
                      summary=summary, files=files, files_sha256=voxcpm_data._digest_json(files))
        (stage / REPORT).write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
        stage.rename(output)
        print(json.dumps(summary, indent=2), flush=True)
        return output / REPORT


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path(voxcpm_data.FAIR_OUTPUT))
    parser.add_argument('--output', type=Path, default=Path(OUTPUT))
    args = parser.parse_args()
    print(prepare(args.source, args.output))

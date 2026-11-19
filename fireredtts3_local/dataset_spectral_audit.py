"""Find persistent and late spectral dips in the actual FireRed training and validation recordings."""
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
import tempfile

import numpy as np
import soundfile as sf

from . import vox_recipe_levels
from .workspace_paths import local_path


# Use equal fractional bandwidths for the lower flank, centre, and upper flank.
FOCUS = {'notch_4k': (3850, 4150), 'broad_4k': (3400, 4600)}
GRID = {f'octave3_{center}': (center / 2**(1/6), center * 2**(1/6))
        for center in (315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000,
                       5000, 6300, 8000, 10000)}
BANDS = {**FOCUS, **GRID}
DEFAULT_OUTPUT = Path('training/diagnostics/dataset-spectral-audit')


def window_spectra(samples, rate):
    """Mean relative power spectra in half-second blocks of active frames."""
    x = np.asarray(samples, dtype=np.float64)
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all() or rate <= 0:
        raise ValueError('Expected finite nonempty mono audio')
    width, hop = max(2, round(.05*rate)), max(1, round(.025*rate))
    nfft = 1 << (width-1).bit_length()
    freq = np.fft.rfftfreq(nfft, 1/rate)
    spectra, times = [], []
    if len(x) >= width:
        frames = np.lib.stride_tricks.sliding_window_view(x, width)[::hop]
        rms_db = 10*np.log10(np.maximum(np.einsum('ij,ij->i', frames, frames)/width, 1e-16))
        active = rms_db > max(-60., np.percentile(rms_db, 90) - 25.)
        window = np.hanning(width)
        for start in range(0, len(frames), 20):
            selected = np.flatnonzero(active[start:start+20]) + start
            if len(selected) < 4:
                continue
            power = np.abs(np.fft.rfft(frames[selected] * window, nfft))**2
            totals = power.sum(axis=1, keepdims=True)
            spectra.append((power / np.maximum(totals, 1e-30)).mean(axis=0))
            times.append((selected.mean()*hop + width/2)/rate)
    return freq, np.array(times), np.asarray(spectra).reshape(-1, len(freq))


def measure(samples, rate):
    freq, times, spectra = window_spectra(samples, rate)
    seconds = len(samples)/rate
    results = {}
    for band, (low, high) in BANDS.items():
        edges = [low*low/high, low, high, high*high/low]
        indices = np.searchsorted(freq, edges)
        row = dict(status='outside_bandwidth', valid_windows=0, median_db=None, head_db=None,
                   tail_db=None, tail_minus_head_db=None, fraction_over_6db=None, strongest_s=None,
                   strongest_db=None, sustained_start_s=None, persistent=False, late=False)
        results[band] = row
        if edges[-1] > rate/2 or min(np.diff(indices)) < 2:
            continue
        row['status'] = 'insufficient_evidence'
        # Require energy on BOTH sides. A bandwidth cutoff alone is not a local notch.
        left, middle, right = [spectra[:, a:b] for a, b in zip(indices[:-1], indices[1:])]
        valid = (left.sum(axis=1) >= 1e-5) & (right.sum(axis=1) >= 1e-5)
        row['valid_windows'] = int(valid.sum())
        if row['valid_windows'] < 3:
            continue
        flank = np.minimum(left[valid].mean(axis=1), right[valid].mean(axis=1))
        depth = 10*np.log10(flank / np.maximum(middle[valid].mean(axis=1), 1e-20))
        t = times[valid]
        head, tail = depth[t < seconds/3], depth[t >= seconds*2/3]
        fraction = float((depth >= 6).mean())
        row.update(status='measured', median_db=float(np.median(depth)), fraction_over_6db=fraction,
                   strongest_s=float(t[depth.argmax()]), strongest_db=float(depth.max()))
        block_seconds = 20 * max(1, round(.025*rate)) / rate
        deep = (t[depth >= 10]/block_seconds).astype(int)
        starts = np.flatnonzero((np.diff(deep)[:-1] == 1) & (np.diff(deep)[1:] == 1))
        if len(starts):
            row['sustained_start_s'] = float(deep[starts[0]]*block_seconds)
        if len(head) >= 2:
            row['head_db'] = float(np.median(head))
        if len(tail) >= 2:
            row['tail_db'] = float(np.median(tail))
        if len(head) >= 2 and len(tail) >= 2:
            delta = row['tail_db'] - row['head_db']
            row.update(tail_minus_head_db=delta, late=row['tail_db'] >= 6 and delta >= 4,
                       persistent=fraction >= .65 and min(row['head_db'], row['tail_db']) >= 6)
    return results


def scan(path):
    samples, rate = sf.read(path, dtype='float32')
    return measure(samples, rate), rate, len(samples)/rate


def inventory(splits, draws):
    """Keep one row per split/corpus/utterance; count target and reference exposure separately."""
    records = {}
    for split, pairs in splits.items():
        for index, (ref, target, _) in enumerate(pairs):
            for role, item in [('target', target), ('reference', ref)]:
                key = (split, item['dataset_id'], item['utt_id'])
                if key not in records:
                    records[key] = (item, Counter())
                if split == 'train' and index < draws:
                    records[key][1][role] += 1
    return records


CLIP_FIELDS = ('split', 'corpus', 'speaker', 'utt_id', 'audio', 'source_audio', 'sample_rate', 'seconds',
               'target_draws_500', 'reference_draws_500', 'measured_bands', 'persistent_bands', 'late_bands',
               'sustained_bands')
BAND_FIELDS = ('band', 'low_hz', 'high_hz', 'status', 'valid_windows', 'median_db', 'head_db', 'tail_db',
               'tail_minus_head_db', 'fraction_over_6db', 'strongest_s', 'strongest_db', 'sustained_start_s',
               'persistent', 'late')
COMPARISON = ('status', 'median_db', 'tail_db', 'tail_minus_head_db', 'sustained_start_s')


def csv_file(path, fields, rows):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def spectrograms(folder, candidates, limit):
    """Plot the same active half-second spectra used for scoring, with silent windows left blank."""
    from PIL import Image, ImageDraw
    saved = set()
    for row in candidates:
        audio = row['audio']
        if audio in saved:
            continue
        if len(saved) >= limit:
            continue
        x, sr = sf.read(audio, dtype='float32')
        freq, times, spectra = window_spectra(x, sr)
        block_seconds = 20 * max(1, round(.025*sr)) / sr
        duration = len(x)/sr
        ceiling = min(sr/2, max(6500., row['high_hz']*1.5))
        grid = np.linspace(ceiling, 0, 240)
        tile = np.full((240, int(np.ceil(duration/block_seconds))), 255, dtype=np.uint8)
        for t, shape in zip(times, spectra):
            density_db = 10*np.log10(np.maximum(np.interp(grid, freq, shape)/(freq[1]-freq[0]), 1e-16))
            tile[:, int(t/block_seconds)] = (np.clip((density_db+90)/60, 0, 1)*220).astype('uint8')
        image = Image.new('RGB', (860, 317), 'white')
        left, top, width, height = 53, 32, 790, 240
        columns = ((np.arange(width)+.5)*duration/width/block_seconds).astype(int)
        image.paste(Image.fromarray(tile[:, columns]), (left, top))
        draw = ImageDraw.Draw(image)
        draw.text((left, 7), f"{row['band']}  |  relative spectrum -90..-30 dB/Hz (dark = weak, white = unmeasured)",
                  fill='black')
        for hz in range(0, int(ceiling)+1, 1000):
            draw.text((5, top+height*(1-hz/ceiling)-6), f'{hz}', fill='black')
        for hz in (row['low_hz'], row['high_hz']):
            y = top+height*(1-hz/ceiling)
            draw.line((left, y, left+width-1, y), fill='#dc7743')
        for frac in (0., 1/3, 2/3, 1.):
            pos = left+frac*(width-1)
            draw.text((pos-10, top+height+5), f'{frac*duration:.1f}', fill='black')
        draw.text((left, 296), 'seconds     |     vertical axis: Hz     |     orange: tested band', fill='black')
        dest = Path('spectrograms')/f'candidate-{len(saved)+1:02d}.png'
        (folder/dest.parent).mkdir(exist_ok=True)
        image.save(folder/dest)
        saved.add(audio)
        row['spectrogram'] = str(dest)


def analyze(dataset, output, plot_top=5):
    from .vox_recipe_transfer import ACCUM, MICRO, paired_data
    dataset, output = local_path(dataset).resolve(), output.resolve()
    if plot_top < 0:
        raise ValueError('--plot-top must be nonnegative')
    if output.exists():
        raise FileExistsError(f'Choose a new report directory: {output}')
    print(f'[audit] Reading the same audio selection as the trainer: {dataset}', flush=True)
    splits, _, selection = paired_data(dataset)
    records = inventory(splits, 500*MICRO*ACCUM)
    output.parent.mkdir(parents=True, exist_ok=True)
    clips, candidates = [], []
    groups, totals = defaultdict(Counter), defaultdict(Counter)
    with tempfile.TemporaryDirectory(dir=output.parent) as tmp:
        stage = Path(tmp)/'report'
        stage.mkdir()
        with (stage/'bands.csv').open('w', encoding='utf-8', newline='') as stream:
            fields = ('split', 'corpus', 'speaker', 'utt_id', *BAND_FIELDS)
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for index, ((split, corpus, uid), (item, counts)) in enumerate(sorted(records.items()), 1):
                paths = {}
                if 'unlevelled_audio' in item:
                    paths['before_levels'] = local_path(item['unlevelled_audio'])
                if 'vad_pcm_sha256' in item:
                    paths['before_denoise'] = local_path(item['trim']['destination'])
                bands, rate, seconds = scan(item['audio'])
                flags = [name for name, b in bands.items() if b['persistent'] or b['late']
                         or b['sustained_start_s'] is not None]
                prev = {name: scan(path)[0] for name, path in paths.items()} if flags else {}
                key = dict(split=split, corpus=corpus, speaker=item['speaker_id'], utt_id=uid)
                clip = dict(**key, audio=item['audio'], source_audio=str(local_path(item['source_audio'])),
                            sample_rate=rate, seconds=seconds, target_draws_500=counts['target'],
                            reference_draws_500=counts['reference'],
                            measured_bands=sum(b['status'] == 'measured' for b in bands.values()),
                            persistent_bands=' '.join(n for n in flags if bands[n]['persistent']),
                            late_bands=' '.join(n for n in flags if bands[n]['late']),
                            sustained_bands=' '.join(n for n in flags if bands[n]['sustained_start_s'] is not None))
                clips.append(clip)
                group = groups[split, corpus]
                group.update(clips=1, measurable=int(clip['measured_bands'] > 0), candidates=int(bool(flags)),
                             measurable_4k=int(any(bands[n]['status'] == 'measured' for n in FOCUS)),
                             candidates_4k=int(any(n in FOCUS for n in flags)), target_draws_500=counts['target'],
                             candidate_target_draws_500=counts['target'] if flags else 0)
                for name, b in bands.items():
                    info = dict(band=name, low_hz=BANDS[name][0], high_hz=BANDS[name][1], **b)
                    writer.writerow({**key, **info})
                    totals[split, corpus, name].update(clips=1, measured=int(b['status'] == 'measured'),
                            tail_comparable=int(b['tail_minus_head_db'] is not None),
                            persistent=int(b['persistent']), late=int(b['late']),
                            sustained=int(b['sustained_start_s'] is not None))
                    if name in flags:
                        comparison = {}
                        for label, measured in prev.items():
                            comparison[label+'_audio'] = str(paths[label])
                            comparison.update({label+'_'+field: measured[name][field] for field in COMPARISON})
                        candidates.append({**clip, **info, **comparison})
                if index % 50 == 0 or index == len(records):
                    print(f'[audit] {index}/{len(records)} recordings', flush=True)
        candidates.sort(key=lambda r: (r['band'] not in FOCUS, not r['late'],
                                       -max(v for v in (r['median_db'], r['tail_db']) if v is not None)))
        comparison_fields = [name+'_'+field for name in ('before_levels', 'before_denoise')
                             for field in ('audio', *COMPARISON)]
        if plot_top:
            spectrograms(stage, candidates, plot_top)
        csv_file(stage/'clips.csv', CLIP_FIELDS, clips)
        csv_file(stage/'candidates.csv', (*CLIP_FIELDS, *BAND_FIELDS, *comparison_fields, 'spectrogram'), candidates)
        summary = dict(dataset=str(dataset), selection_sha256=selection['selection_sha256'],
                       recordings=len(clips), candidate_recordings=sum(g['candidates'] for g in groups.values()),
                       groups=[dict(split=s, corpus=c, **v) for (s, c), v in sorted(groups.items())],
                       bands=[dict(split=s, corpus=c, band=n, **v) for (s, c, n), v in sorted(totals.items())],
                       method='local-notch-v1', depth_db=6, minimum_fraction=.65, tail_increase_db=4)
        (stage/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
        stage.rename(output)
    print(f'[audit] {summary["candidate_recordings"]}/{len(clips)} candidates. Reports: {output}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=Path(vox_recipe_levels.OUTPUT))
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--plot-top', type=int, default=5, help='Spectrograms of the first N distinct candidates')
    args = parser.parse_args()
    analyze(args.dataset, args.output, args.plot_top)

"""Measure long native FLOAT WAVs and make RMS-matched listening controls (NumPy/SciPy/Matplotlib)."""
import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.io import wavfile


# Also support the documented standalone command in an isolated plotting environment.
if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fireredtts3_local.tail_contrast import BANDS, frame_contrast


def db(power):
    return 10 * np.log10(np.maximum(power, 1e-20))


def measure(path):
    sr, x = wavfile.read(path)
    if sr != 24000 or x.ndim != 1 or x.dtype.kind != 'f' or len(x) < 6*sr or not np.isfinite(x).all():
        raise ValueError(f'Expected a finite native 24 kHz mono FLOAT WAV of at least 6 seconds: {path}')
    x = x.astype(np.float64)
    f, t, stft = signal.stft(x, sr, nperseg=1920, noverlap=1680, nfft=4096,
                             boundary=None, padded=False, scaling='psd')
    power = abs(stft)**2
    power[1:-1] *= 2
    frame_power = power.sum(axis=0)*f[1]
    active = db(frame_power) > -40
    seconds = len(x)/sr
    head = active & (t < seconds/3)
    tail = active & (t >= 2*seconds/3)
    if not head.any() or not tail.any():
        raise ValueError(f'No active frames in first or last third: {path}')
    n = min(10*sr, len(x)//3)
    row = dict(file=path.name, seconds=seconds, rms_window_s=n/sr,
               tail_minus_head_rms_db=db(np.mean(x[-n:]**2))-db(np.mean(x[:n]**2)),
               sample_peak_dbfs=db(abs(x).max()**2), pct_samples_ge_1=100*np.mean(abs(x)>=1))
    # Give each active frame equal weight and cancel its broad-band level.
    shapes = [db((power[:, m]/frame_power[m]).mean(axis=1)) for m in (head, tail)]
    trough = (3400 <= f) & (f <= 4600)
    band_power = [np.mean(10**(shape[trough]/10)) for shape in shapes]
    row['relative_3400_4600_delta_db'] = db(band_power[1])-db(band_power[0])
    for fc, w in BANDS:
        distance = np.abs(f-fc)
        inside = distance <= w
        peak = np.flatnonzero(inside)[np.argmax(shapes[1][inside])]
        row[f'peak_{fc}_hz'] = f[peak]
        row[f'at_peak_{fc}_normalized_delta_db'] = shapes[1][peak]-shapes[0][peak]
    contrast = frame_contrast(f, power)
    row['head_contrast_db'] = np.median(contrast[head])
    row['tail_contrast_db'] = np.median(contrast[tail])
    bins = []
    for i in range(int(seconds)):
        frames = active & (t >= i) & (t < i+1)
        bins.append(np.median(contrast[frames]) if frames.sum() >= 10 else np.nan)
    bins = np.array(bins)
    baseline = np.nanmedian(bins[:5])
    row['sustained_plus6_start_s'] = next((i for i in range(5, len(bins)-2)
                                           if np.all(bins[i:i+3] > baseline+6)), None)
    return row, f, t, power, shapes, bins


def bell(x, sr, hz, q, gain_db):
    a = 10**(gain_db/40)
    w = 2*np.pi*hz/sr
    k = np.sin(w)/(2*q)
    return signal.lfilter([1+k*a, -2*np.cos(w), 1-k*a], [1+k/a, -2*np.cos(w), 1-k/a], x)


def listening_pair(path, out):
    sr, x = wavfile.read(path)
    x = x.astype(float)
    # Filter the entire file before cutting the excerpt to preserve filter state.
    filtered = x
    for fc, q in [(2220, 22), (2965, 27), (5190, 24)]:
        filtered = bell(filtered, sr, fc, q, -12)
    lifted = bell(filtered, sr, 4000, 3, 8)
    clips = [wave[-8*sr:] for wave in (x, filtered, lifted)]
    target = np.mean(clips[0]**2)
    clips = [clip*np.sqrt(target/max(np.mean(clip**2), 1e-20)) for clip in clips]
    gain = .94/max(max(abs(clip).max() for clip in clips), 1e-10)
    for label, y in zip(['original', 'reduced', 'rebalanced'], clips):
        wavfile.write(out/f'{path.stem}-{label}.wav', sr, np.rint(y*gain*32767).astype('int16'))


def analyze(audio, out):
    paths = sorted(audio.glob('*.wav'))
    if not paths:
        raise ValueError(f'No WAVs in {audio}')
    out.mkdir(parents=True, exist_ok=False)
    ab = out/'AB'
    ab.mkdir()
    rows, shapes, tracks = [], [], []
    nrows = (len(paths)+1)//2
    fig, axes = plt.subplots(nrows, 2, figsize=(14, 1.95*nrows+1.3), squeeze=False,
                             layout='constrained', sharex=True, sharey=True)
    for ax, path in zip(axes.flat, paths):
        row, f, t, p, spectra, bins = measure(path)
        rows.append(row)
        shapes.append(spectra)
        tracks.append(bins)
        # Average power over 40 ms for display only; analysis above uses all frames.
        n = (len(t)//4)*4
        fr = (f>=75)&(f<=11500)
        tile = p[fr,:n].reshape(fr.sum(), -1, 4).mean(axis=2)
        im = ax.pcolormesh(t[:n].reshape(-1,4).mean(1), f[fr], db(tile), shading='auto',
                           vmin=-100, vmax=-42, cmap='magma', rasterized=True)
        ax.set_yscale('log')
        ax.set_ylim(80, 11300)
        ax.set_yticks([125, 500, 2000, 8000], labels=['125', '500', '2k', '8k'])
        for mult in [1/3, 2/3]:
            ax.axvline(row['seconds']*mult, color='cyan', lw=.8, ls=':')
        ax.set_title(f"{path.stem}   {row['seconds']:.2f} с", fontsize=10, loc='left')
        listening_pair(path, ab)
    for ax in list(axes.flat)[len(paths):]:
        ax.set_visible(False)
    fig.suptitle('Длинные реплики · спектрограммы исходных WAV на общей шкале\nПунктир делит каждую запись на трети')
    fig.supxlabel('Время, с')
    fig.supylabel('Частота, Гц')
    fig.colorbar(im, ax=axes.ravel().tolist(), label='дБ от FS²/Гц', shrink=.8)
    fig.savefig(out/'01-spectrograms.png', dpi=145)
    plt.close(fig)
    fig, (ax, bx) = plt.subplots(2, figsize=(11.5, 8), layout='constrained')
    med = np.median(np.stack(shapes), axis=0)
    for shape, title, c in zip(med, ['Первая треть', 'Последняя треть'], ['#346a9a','#b44624']):
        ax.plot(f, shape, label=title, color=c, lw=1)
    for fc, w in BANDS:
        ax.axvspan(fc-w, fc+w, alpha=.13, color='#664da5')
    ax.set(xlim=(1000,6100), ylim=(-83,-25), xlabel='Частота, Гц', ylabel='Относительная плотность, дБ/Гц',
           title='Форма спектра · кадры речи приведены к одной мощности · медиана по файлам')
    ax.legend()
    for row, bins in zip(rows, tracks):
        bx.plot(np.arange(len(bins))+.5, bins, lw=1.3, label=Path(row['file']).stem)
    bx.set(xlabel='Время, с', ylabel='Контраст полос, дБ', title='Три узкие полосы относительно соседних частот · по секундам')
    bx.legend(ncol=2, fontsize=8)
    for a in [ax,bx]:
        a.grid(alpha=.2)
    fig.savefig(out/'02-resonances.png', dpi=150)
    plt.close(fig)
    with (out/'measurements.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('audio', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    analyze(args.audio, args.output)

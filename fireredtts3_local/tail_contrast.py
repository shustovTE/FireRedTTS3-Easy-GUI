"""Contrast of the three empirically observed narrow bands on native 24 kHz WAVs."""
import numpy as np

BANDS = [(2220, 85), (2965, 80), (5190, 160)]


def frame_contrast(f, power):
    contrasts = []
    for center, width in BANDS:
        distance = np.abs(f - center)
        mid = power[distance <= width].mean(axis=0)
        flank = power[(distance >= 1.5*width) & (distance <= 2.5*width)].mean(axis=0)
        contrasts.append(10*np.log10(np.maximum(mid, 1e-20)/np.maximum(flank, 1e-20)))
    return np.median(contrasts, axis=0)


def contrast_windows(samples, rate):
    if rate != 24000:
        raise ValueError('Tail resonance measurements require native 24 kHz output')
    # Match scipy.signal.stft in inspect_tail_resonances, without requiring SciPy on Gauss.
    width, hop, nfft = 1920, 240, 4096
    t = np.arange(0, len(samples)-width+1, hop)/rate + width/(2*rate)
    contrast = np.full(len(t), np.nan)
    if len(t):
        window = np.hanning(width+1)[:-1]
        frames = np.lib.stride_tricks.sliding_window_view(samples, width)[::hop]
        f = np.fft.rfftfreq(nfft, 1/rate)
        for i in range(0, len(t), 128):
            x = frames[i:i+128]*window
            power = abs(np.fft.rfft(x, nfft, axis=1).T)**2/(rate*(window@window))
            power[1:-1] *= 2
            active = power.sum(axis=0)*f[1] > 1e-4
            contrast[i:i+len(x)] = np.where(active, frame_contrast(f, power), np.nan)
    def median(start, end):
        v = contrast[(t >= start) & (t < end) & np.isfinite(contrast)]
        return float(np.median(v)) if len(v) else None
    seconds = len(samples)/rate
    return dict(head_contrast_db=median(0, seconds/3), tail_contrast_db=median(2*seconds/3, seconds),
                contrast_15_19_db=median(15, 19) if seconds >= 19 else None)

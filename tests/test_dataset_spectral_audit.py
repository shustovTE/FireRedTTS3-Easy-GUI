import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf

from fireredtts3_local import dataset_spectral_audit as audit


def filtered(x, rate, low, high):
    f = np.fft.rfftfreq(len(x), 1/rate)
    spectrum = np.fft.rfft(x)
    spectrum[(low <= f) & (f < high)] *= .10
    return np.fft.irfft(spectrum, n=len(x))


class SpectralAuditTests(unittest.TestCase):
    def test_global_tail_and_mid_notches_and_gain_invariance(self):
        rate = 16000
        x = np.random.default_rng(42).normal(0, .07, 12*rate)
        def band(y, name='notch_4k'):
            return audit.measure(y, rate)[name]
        self.assertFalse(band(x)['persistent'])
        narrow = filtered(x, rate, 3850, 4150)
        entire = band(narrow)
        self.assertTrue(entire['persistent'])
        self.assertFalse(entire['late'])
        self.assertTrue(band(filtered(x, rate, 3400, 4600), 'broad_4k')['persistent'])
        self.assertTrue(band(filtered(x, rate, *audit.GRID['octave3_1250']), 'octave3_1250')['persistent'])
        tail = band(np.r_[x[:8*rate], narrow[8*rate:]])
        self.assertTrue(tail['late'])
        self.assertFalse(tail['persistent'])
        self.assertGreater(tail['tail_minus_head_db'], 8)
        mid = band(np.r_[x[:4*rate], narrow[4*rate:8*rate], x[8*rate:]])
        self.assertFalse(mid['late'] or mid['persistent'])
        self.assertAlmostEqual(mid['sustained_start_s'], 4, delta=.5)
        self.assertAlmostEqual(entire['median_db'], band(narrow*.25)['median_db'])
        gain_only = band(x * np.linspace(.5, 1.8, len(x)))
        self.assertFalse(gain_only['late'] or gain_only['persistent'])

    def test_silence_lowpass_gaps_and_native_nyquist(self):
        self.assertEqual(audit.measure(np.zeros(5), 16000)['notch_4k']['status'], 'insufficient_evidence')
        x = np.random.default_rng(7).normal(0, .08, 8*16000)
        self.assertEqual(audit.measure(x, 8000)['notch_4k']['status'], 'outside_bandwidth')
        spectrum = np.fft.rfft(x)
        spectrum[np.fft.rfftfreq(len(x), 1/16000) > 3900] = 0
        self.assertEqual(audit.measure(np.fft.irfft(spectrum), 16000)['notch_4k']['status'], 'insufficient_evidence')
        # Three deep but isolated active blocks cannot establish a 1.5s event.
        burst = filtered(x[:6400], 16000, 3850, 4150)
        intermittent = np.zeros_like(x)
        for start in (16000, 48000, 80000):
            intermittent[start:start+6400] = burst
        self.assertIsNone(audit.measure(intermittent, 16000)['notch_4k']['sustained_start_s'])

    def test_reports_unique_clips_exposure_and_preprocessing_stages(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rate = 16000
            dry = np.random.default_rng(21).normal(0, .08, 6*rate)
            wet = filtered(dry, rate, 3850, 4150)
            original, enhanced, final = [root/(s+'.wav') for s in ('source', 'enhanced', 'normalized')]
            for p, y in zip((original, enhanced, final), (dry, wet, wet*.7), strict=True):
                sf.write(p, y, rate, subtype='FLOAT')
            a = dict(utt_id='a', dataset_id='spontaneous', speaker_id='voice', audio=str(final),
                     source_audio=str(original), unlevelled_audio=str(enhanced), vad_pcm_sha256='marked',
                     trim={'destination': str(original)})
            b = dict(a, utt_id='b', audio=str(original), unlevelled_audio=str(original))
            splits = {'train': [(a, b, 0)]*4 + [(b, a, 0)]*2, 'validation': [(b, a, 0)]}
            with patch('fireredtts3_local.vox_recipe_transfer.paired_data',
                       return_value=(splits, [], {'selection_sha256': 'abc'})):
                out = root/'out'
                audit.analyze(root, out)
            summary = json.loads((out/'summary.json').read_text())
            self.assertEqual(summary['recordings'], 4)
            train = next(g for g in summary['groups'] if g['split'] == 'train')
            self.assertEqual((train['candidates_4k'], train['target_draws_500']), (1, 6))
            with (out/'candidates.csv').open() as f:
                row = next(r for r in csv.DictReader(f) if r['split']=='train' and r['band']=='notch_4k')
            self.assertEqual((row['target_draws_500'], row['reference_draws_500']), ('2', '4'))
            self.assertGreater(float(row['before_levels_median_db']), 6)
            self.assertLess(float(row['before_denoise_median_db']), 2)
            self.assertTrue(any((out/'spectrograms').glob('*.png')))
            self.assertTrue((out/'bands.csv').exists())

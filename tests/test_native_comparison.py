"""Keep diagnostic loudness growth visible, even above PCM full scale."""
import unittest

import numpy as np

from fireredtts3_local.native_comparison import signal_levels


class SignalLevelsTests(unittest.TestCase):
    def test_amplitude_growth_and_over_range_samples_are_not_normalized(self):
        wave = np.concatenate([np.full(100, .1), np.full(100, 1.2), np.zeros(50)])
        original = wave.copy()
        levels = signal_levels(wave, 100)
        self.assertEqual(levels["seconds"], 2.5)
        self.assertEqual(levels["peak"], 1.2)
        self.assertEqual(levels["fraction_at_or_above_full_scale"], .4)
        self.assertAlmostEqual(levels["windows"][0]["rms_dbfs"], -20.)
        self.assertGreater(levels["windows"][1]["rms_dbfs"], 0.)
        self.assertEqual(levels["windows"][2]["seconds"], .5)
        np.testing.assert_array_equal(wave, original)

    def test_invalid_audio_does_not_produce_misleading_metrics(self):
        for wave in ([], [float("nan")], [float("inf")], [[.1, .2]]):
            with self.assertRaises(ValueError):
                signal_levels(wave, 24000)


if __name__ == "__main__":
    unittest.main()

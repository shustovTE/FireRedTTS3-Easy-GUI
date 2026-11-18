import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import soundfile as sf
import torch

from fireredtts3_local import vox_recovery
from fireredtts3_local.training_batch import Example


class AcousticRecoveryTests(unittest.TestCase):
    def test_prefix_excludes_future_and_uses_native_left_padding(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / 'source.wav'
            wav = np.ones(91, dtype=np.float32) * .2
            # At 8 kHz this is 46 samples, left padded with 18 to the 32-sample grid.
            sf.write(src, wav, 16000, subtype='FLOAT')
            first = vox_recovery.read_prefix(src, 32, 64, 32, 8000)
            wav[28:] = -1
            sf.write(src, wav, 16000, subtype='FLOAT')
            again = vox_recovery.read_prefix(src, 32, 64, 32, 8000)
            torch.testing.assert_close(first, again, rtol=0, atol=0)
            self.assertEqual(first.shape, (1, 32))
            self.assertFalse(first[:, :18].any())
            self.assertGreater(first[:, 18:].sum(), 0)
            with self.assertRaisesRegex(ValueError, 'grid'):
                vox_recovery.read_prefix(src, 32, 96, 32, 8000)

    def test_flow_target_is_clean_and_history_is_reencoded(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp)/'src.wav')
            wave = np.sin(np.arange(60)*.7).astype('float32')
            sf.write(path, wave, 1000, subtype='FLOAT')
            encode = lambda w, rate: w.reshape(1, -1, 2)
            z = encode(torch.from_numpy(wave), 1000)
            ref = z[:, :2].clone()
            example = Example(torch.tensor([1]), ref, z, torch.ones(1, 2),
                              prompt_embeds=torch.ones(1, 1, 3), target_embeds=torch.ones(1, 15, 3))
            core = SimpleNamespace(patch_size=2)
            tts = SimpleNamespace(tts_core=core, redae=SimpleNamespace(sample_rate=1000, downsample_rate=2, encode=encode))
            lengths = []
            def condition(_, prefixes):
                (e,) = prefixes
                n = e.target.shape[1]
                lengths.append(n)
                self.assertIsNone(e.target_embeds)
                self.assertIs(e.prompt, example.prompt)
                self.assertIs(e.speaker, example.speaker)
                self.assertFalse(torch.equal(e.target, z[:, :n]))
                self.assertFalse(e.target.requires_grad)
                return None, None, e.target[:, -2:], None
            def predict(_, cond, speaker, tensors):
                clean, history, noise, time, drop = tensors
                n = lengths[-1]
                torch.testing.assert_close(clean, z[:, n:n+2], rtol=0, atol=0)
                self.assertFalse(drop.any())
                return (clean - noise) + 1.
            with patch.object(vox_recovery, 'prefix_conditioning', condition), patch.object(vox_recovery, 'predict', predict):
                loss, n = vox_recovery.recovery_loss(tts, [example], [{'audio': path}], 123)
                self.assertEqual(n, 1)
                self.assertAlmostEqual(float(loss), 1., places=5)
                self.assertTrue(14 <= lengths[0] < z.shape[1])

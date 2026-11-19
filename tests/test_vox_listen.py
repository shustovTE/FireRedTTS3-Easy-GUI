import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

from fireredtts3_local import vox_listen


class ListeningTest(unittest.TestCase):
    def test_gated_weights_unsplit_requests_and_unmodified_audio(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / 'lora/step-0500'
            checkpoint.mkdir(parents=True)
            ref = root / 'voice.wav'
            ref.write_bytes(b'fixture audio')
            (checkpoint / 'adapter_model.safetensors').write_bytes(b'fixture adapter')
            control = dict(id='cities_seed1237', reference_audio=str(ref), reference_text='Тихий вечер.',
                           reference_sha256=vox_listen.file_digest(ref))
            gate = dict(percent=20, cases=[control], settings=vox_listen.SETTINGS)
            record = dict(step=500, recipe=dict(rise_gate=gate, mode='lora', base_sha256='base'),
                          validation=dict(accepted_updates=77))
            (checkpoint / 'checkpoint.json').write_text(json.dumps(record))
            output = root / 'wav'
            # Deliberately short, over-full-scale output; retain it as generated.
            audio = np.linspace(-1.25, 1.25, 1000, dtype=np.float32)
            with (patch.object(vox_listen, 'RUN', root), patch.object(vox_listen, 'model_identity', return_value='base'),
                  patch.object(vox_listen, 'ModelManager') as manager,
                  patch.object(vox_listen, 'generate_case', return_value=(audio, 500, {'finish_reason': 'stop'})) as gen):
                vox_listen.listen(output, [1234, 1235])
                manager.return_value.load_base.assert_called_once_with(attention_backend='sdpa', adapter_path=str(checkpoint))
                manager.return_value.unload.assert_called_once()
                with self.assertRaises(FileExistsError):
                    vox_listen.listen(output, [1234, 1235])
            report = json.loads((output / 'report.json').read_text())
            phrases = json.loads(vox_listen.PHRASES.read_text())
            expected = [(s, seed) for group in phrases.values() for s in group.values() for seed in (1234, 1235)]
            self.assertTrue(report['complete'])
            self.assertEqual(report['accepted_updates'], 77)
            self.assertEqual([(c.args[1]['text'], c.args[1]['seed']) for c in gen.call_args_list], expected)
            for call, result in zip(gen.call_args_list, report['results'], strict=True):
                self.assertFalse(call.kwargs['settings']['do_split'])
                self.assertEqual(call.args[1]['reference_audio'], str(ref))
                self.assertEqual(result['seconds'], 2)
                if result['wav'].startswith('long/'):
                    self.assertIs(result['in_target_range'], False)
                samples, _ = sf.read(output / result['wav'], dtype='float32')
                np.testing.assert_array_equal(samples, audio)

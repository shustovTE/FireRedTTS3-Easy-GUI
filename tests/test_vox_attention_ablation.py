import csv
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from fireredtts3_local import vox_attention_ablation as ablate
from fireredtts3_local.vox_recipe_transfer import configure
from test_training_batch import tiny_core


class AblationTests(unittest.TestCase):
    def test_routes_only_the_requested_lora_deltas_and_restores_both(self):
        core = configure(tiny_core(), 'lora')
        layers = dict(core.get_base_model().named_modules())
        names = ['backbone_llm.layers.0.self_attn.q_proj', 'dit.blocks.0.attn.to_q']
        with torch.no_grad():
            for name in names:
                layers[name].lora_A['default'].weight.fill_(.1)
                layers[name].lora_B['default'].weight.fill_(.2)
            first = []
            for mode in ('both', 'backbone', 'dit', 'both'):
                counts = ablate.route(core, mode)
                for i, (name, group) in enumerate(zip(names, ('backbone', 'dit'))):
                    layer = layers[name]
                    x = torch.ones(1, layer.in_features)
                    out = layer(x)
                    enabled = mode in (group, 'both')
                    self.assertEqual(counts[group] > 0, enabled)
                    self.assertEqual(torch.equal(out, layer.base_layer(x)), not enabled)
                    if len(first) < 2:
                        first.append(out.clone())
                    if enabled:
                        torch.testing.assert_close(out, first[i], atol=0., rtol=0.)
                    self.assertFalse(any(p.requires_grad for p in layer.parameters()))

    def test_paired_checkpoints_shared_base_restart_and_short_audio(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp)
            output = parent/'matrix'
            ref = parent/'reference.wav'
            ref.write_bytes(b'original reference')
            fixtures = ((parent/'current', 100, ablate.vox_recovery.RESONANT),
                        (parent/'prev', 500, ablate.vox_recovery.FOUR_PATCH))
            for run, step, profile in fixtures:
                path = run/'lora'/f'step-{step:04d}'
                path.mkdir(parents=True)
                (path/'adapter_config.json').write_text('{}')
                (path/'adapter_model.safetensors').write_bytes(str(step).encode())
                gate = dict(percent=20, settings=ablate.SETTINGS, cases=[dict(id='cities_seed1237',
                    reference_audio=str(ref), reference_text='Да.', reference_sha256=ablate.file_digest(ref))])
                (path/'checkpoint.json').write_text(json.dumps(dict(step=step, validation=dict(accepted_updates=77),
                    recipe=dict(mode='lora', rise_gate=gate, dataset_sha256='data', base_sha256=ablate.BASE_SHA,
                                acoustic_recovery=profile))))
            generated = []
            def gen(tts, row, settings):
                self.assertFalse(settings['do_split'])
                self.assertEqual(row['reference_audio'], str(ref))
                generated.append((row['text'], row['seed']))
                return np.ones(3000, dtype=np.float32)*1.1, 24000, {'finish_reason':'stop'}
            with (patch.object(ablate, 'model_identity', return_value=ablate.BASE_SHA),
                  patch.object(ablate, 'ModelManager') as manager, patch.object(ablate, 'route', return_value={}),
                  patch.object(ablate, 'generate_case', side_effect=gen)):
                ablate.run(output, parent/'current', parent/'prev', audio_only=True)
                self.assertEqual(manager.return_value.load_base.call_count, 3)
                saved = json.loads((output/'report.json').read_text())
                self.assertTrue(saved['audio_complete'])
                cases = [(c['text'], c['seed']) for c in saved['plan']['cases']]
                self.assertEqual(len(cases), 8)
                self.assertEqual(generated, cases*7)
                ablate.run(output, parent/'current', parent/'prev', audio_only=True)
                self.assertEqual(len(generated), 56)
                name = next(iter(saved['results']))
                (output/name).write_bytes(b'interrupted')
                ablate.run(output, parent/'current', parent/'prev', audio_only=True)
                self.assertEqual(len(generated), 57)
                with (output/'summary.csv').open() as stream:
                    rows = list(csv.DictReader(stream))
                self.assertEqual(len(rows), 56)
                self.assertTrue(all(row['contrast_15_19_db'] == '' for row in rows))
                with patch.object(ablate, 'load_asr') as load:
                    whisper = Mock()
                    load.return_value = (whisper, {})
                    whisper.transcribe.return_value = ([SimpleNamespace(text='Тест.')], None)
                    ablate.run(output, parent/'current', parent/'prev')
                    self.assertEqual(whisper.transcribe.call_count, 56)
                    scored = json.loads((output/'report.json').read_text())
                    self.assertTrue(scored['asr_complete'])
                    self.assertEqual(scored['results'][name]['transcript'], 'Тест.')
                    self.assertEqual(len(generated), 57)
                with self.assertRaisesRegex(ValueError, 'training recipe'):
                    conf = fixtures[0][0]/'lora/step-0100/checkpoint.json'
                    rec = json.loads(conf.read_text())
                    rec['recipe']['acoustic_recovery'] = ablate.vox_recovery.FOUR_PATCH
                    conf.write_text(json.dumps(rec))
                    ablate.run(output, parent/'current', parent/'prev', audio_only=True)

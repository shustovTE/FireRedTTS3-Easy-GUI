import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

from fireredtts3_local import vox_recipe_sweep as replay


class ReplayTest(unittest.TestCase):
    def test_replays_from_pilot_even_with_500_and_recovers_between_audio_series(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, data = root/'original', root/'dataset'
            data.mkdir()
            (data/replay.vox_recipe_levels.REPORT).write_text('{}')
            recipe = dict(mode='lora', base_sha256=replay.BASE_SHA, dataset_sha256='levelled')
            pilot = source/'lora'/'step-0030'
            pilot.mkdir(parents=True)
            (pilot/'checkpoint.json').write_text(json.dumps(dict(step=30, recipe=recipe)))
            for name in ('adapter_config.json', 'adapter_model.safetensors', 'training_state.pt'):
                (pilot/name).write_bytes(name.encode())
            later = source/'lora'/'step-0500'
            later.mkdir()
            (later/'checkpoint.json').write_text('original 500')

            reference = root/'voice.wav'
            sf.write(reference, np.zeros(16000, dtype='float32'), 16000)
            reports = []
            for seed in (1234, 1236):
                folder = source/f'pilot-{seed}'
                folder.mkdir()
                cases = folder/'benchmark.jsonl'
                cases.write_text(json.dumps(dict(id='q', text='Когда?', reference_audio=str(reference),
                                                 reference_text='Добрый день.', seed=seed)))
                _, sha = replay.load_benchmark(cases)
                path = folder/'report.json'
                path.write_text(json.dumps(dict(complete=True, step=30, recipes={'lora': recipe}, benchmark_sha256=sha,
                                                settings=replay.SETTINGS, asr='same-asr', speaker_model='same-voice')))
                reports.append(path)

            updates, generations = [], []
            def train(dataset, output, mode, step, resume, *, fork_from, shared_features):
                self.assertEqual((dataset, mode, resume, shared_features), (data, 'lora', True, source/'features'))
                self.assertEqual(fork_from, pilot if step == 100 else None)
                saved = output/'lora'/f'step-{step:04d}'
                saved.mkdir(parents=True)
                (saved/'checkpoint.json').write_text(json.dumps(dict(step=step, recipe=recipe)))
                updates.append(step)

            interrupted = False
            def compare(dataset, output, step, benchmark, asr, speaker_model, *, against, lora_only):
                nonlocal interrupted
                self.assertTrue(lora_only)
                self.assertEqual((dataset, asr, speaker_model), (data, 'same-asr', 'same-voice'))
                if not interrupted and step == 200 and against == reports[1]:
                    interrupted = True
                    raise RuntimeError('interrupted during evaluation')
                generations.append((step, against))
                folder = Path(tempfile.mkdtemp(dir=output))
                (folder/'lora').mkdir()
                sf.write(folder/'lora'/'case-000.wav', np.zeros(16000), 16000)
                (folder/'report.json').write_text('{"complete":true}')
                return folder/'report.json'

            with patch.object(replay, 'train', train), patch.object(replay, 'compare', compare):
                with self.assertRaisesRegex(RuntimeError, 'interrupted'):
                    replay.sweep(data, source, reports)
                run = replay.sweep(data, source, reports)
                replay.sweep(data, source, reports)
                self.assertEqual(updates, [100, 200, 300, 400])
                self.assertEqual(generations, [(s, p) for s in updates for p in reports])
                for step in updates:
                    for series in (1, 2):
                        self.assertTrue((run/'audio'/f'step-{step:04d}'/f'series-{series}'/'lora'/'case-000.wav').is_file())
                self.assertEqual((later/'checkpoint.json').read_text(), 'original 500')
                self.assertEqual(sorted((source/'lora').iterdir()), [pilot, later])
                (pilot/'adapter_model.safetensors').write_bytes(b'different pilot')
                with self.assertRaisesRegex(ValueError, 'Replay inputs differ'):
                    replay.sweep(data, source, reports)

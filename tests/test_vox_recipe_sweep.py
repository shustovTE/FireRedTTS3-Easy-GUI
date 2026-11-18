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
            def train(dataset, output, mode, step, resume, *, fork_from, shared_features, recovery):
                self.assertEqual((dataset, mode, resume, shared_features), (data, 'lora', True, source/'features'))
                self.assertEqual(fork_from, pilot if step in (50, 100) else None)
                saved = output/'lora'/f'step-{step:04d}'
                saved.mkdir(parents=True)
                actual = dict(recipe, acoustic_recovery=replay.vox_recovery.CONFIG) if recovery else recipe
                (saved/'checkpoint.json').write_text(json.dumps(dict(step=step, recipe=actual)))
                updates.append(step)

            interrupted = False
            def compare(dataset, output, step, benchmark, asr, speaker_model, *, against, lora_only, reuse_base):
                nonlocal interrupted
                self.assertTrue(lora_only)
                self.assertFalse(reuse_base)
                self.assertEqual((dataset, asr, speaker_model), (data, 'same-asr', 'same-voice'))
                if not interrupted and step == 200 and against == reports[1]:
                    interrupted = True
                    raise RuntimeError('interrupted during evaluation')
                generations.append((step, against))
                folder = Path(tempfile.mkdtemp(dir=output))
                (folder/'lora').mkdir()
                sf.write(folder/'lora'/'case-000.wav', np.zeros(16000), 16000)
                previous = json.loads(against.read_text())
                (folder/'report.json').write_text(json.dumps(dict(complete=True, step=step, compared_to=str(against),
                    benchmark_sha256=previous['benchmark_sha256'], recipes=previous['recipes'], asr=asr, speaker_model=speaker_model)))
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
                updates.clear()
                generations.clear()
                short_run = replay.sweep(data, source, reports, until=50)
                replay.sweep(data, source, reports, until=50)
                self.assertEqual(short_run, source/'lora-0030-to-0050')
                self.assertEqual(updates, [50])
                self.assertEqual(generations, [(50, p) for p in reports])
                self.assertTrue((short_run/'audio'/'step-0050'/'series-2'/'lora'/'case-000.wav').is_file())
                recovered = replay.sweep(data, source, reports, until=50, control=short_run)
                replay.sweep(data, source, reports, until=50, control=short_run)
                self.assertEqual(recovered.name, 'lora-0030-to-0050-recovery')
                self.assertEqual(updates, [50, 50])
                self.assertEqual(generations[2:], [(50, short_run/'audio'/'step-0050'/f'series-{i}'/'report.json')
                                                    for i in (1, 2)])
                self.assertTrue((run/'lora'/'step-0100'/'checkpoint.json').is_file())
                (pilot/'adapter_model.safetensors').write_bytes(b'different pilot')
                with self.assertRaisesRegex(ValueError, 'Replay inputs differ'):
                    replay.sweep(data, source, reports)

    def test_fresh_recovery_without_a_pilot_and_restart_after_first_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)
            (source/replay.vox_recipe_levels.REPORT).write_text('{}')
            path = source/'report.json'
            path.write_text(json.dumps(dict(asr='asr', speaker_model='speaker')))
            recipe = dict(replay.transfer_recipe('lora', 'data'), recovery_start_step=0)
            updates, samples = [], []
            def train(dataset, run, mode, step, resume, *, fork_from, shared_features, recovery):
                self.assertEqual((mode, fork_from, resume), ('lora', None, step != 50))
                profile = replay.vox_recovery.STRONGER_COMB if 'comb050' in run.name else replay.vox_recovery.CONFIG
                self.assertEqual(recovery, profile)
                if resume:
                    self.assertTrue((run/'lora'/'step-0050'/'checkpoint.json').is_file())
                checkpoint = run/'lora'/f'step-{step:04d}'
                checkpoint.mkdir(parents=True)
                (checkpoint/'checkpoint.json').write_text(json.dumps(dict(step=step, recipe=dict(recipe, acoustic_recovery=recovery))))
                updates.append(step)

            def compare(dataset, run, step, benchmark, asr, speaker_model, *, against, lora_only, reuse_base):
                self.assertEqual((against, lora_only, reuse_base), (path, True, True))
                samples.append(step)
                if len(samples) == 1:
                    raise RuntimeError('evaluation interrupted')
                folder = Path(tempfile.mkdtemp(dir=run))
                (folder/'lora').mkdir()
                (folder/'lora'/'case-000.wav').touch()
                checkpoint = json.loads((run/'lora'/f'step-{step:04d}'/'checkpoint.json').read_text())
                case = dict(id='cities', text='Вопрос?', seed=1, asr={'wer':0.},
                            end_minus_start_rms_db=3. if 'comb050' in run.name else 4.,
                            voice_similarity=.8, final_voice_similarity=.7, audio='lora/case-000.wav')
                (folder/'report.json').write_text(json.dumps(dict(complete=True, step=step, benchmark_sha256='cases',
                    settings={}, asr=asr, speaker_model=speaker_model, recipes={'lora':checkpoint['recipe']},
                    results={'lora':[case]})))
                return folder/'report.json'

            with patch.object(replay, 'paired_data', return_value=({}, [], {'selection_sha256':'data'})), \
                    patch.object(replay, 'load_saved_base'), patch.object(replay, 'train', train), \
                    patch.object(replay, 'compare', compare):
                with self.assertRaisesRegex(RuntimeError, 'evaluation interrupted'):
                    replay.fresh_recovery(source, source, [path])
                run = replay.fresh_recovery(source, source, [path])
                replay.fresh_recovery(source, source, [path])
                strong = replay.fresh_recovery(source, source, [path], stronger_comb=True)
                old_path = run/'audio'/'step-0050'/'series-1'/'report.json'
                original_report = old_path.read_text()
                changed = json.loads(original_report)
                changed['results']['lora'][0]['seed'] += 1
                old_path.write_text(json.dumps(changed))
                replay.fresh_recovery(source, source, [path], stronger_comb=True)
                incomplete = json.loads((strong/'audio'/'step-0050'/'series-1'/'paired-recovery.json').read_text())
                self.assertFalse(incomplete['complete'])
                self.assertIn('seed mismatch', incomplete['reason'])
                old_path.write_text(original_report)
                replay.fresh_recovery(source, source, [path], stronger_comb=True)
            self.assertNotEqual(strong, run)
            self.assertEqual(updates, [50, 100, 50, 100])
            self.assertEqual(samples, [50, 50, 100, 50, 100])
            for n in (50, 100):
                self.assertTrue((run/'audio'/f'step-{n:04d}'/'series-1'/'lora'/'case-000.wav').is_file())
                paired = json.loads((strong/'audio'/f'step-{n:04d}'/'series-1'/'paired-recovery.json').read_text())
                pair = paired['cases'][0]
                self.assertEqual(pair['delta_vs_standard']['end_minus_start_rms_db'], -1.)
                self.assertTrue(Path(pair['standard_wav']).is_file() and Path(pair['stronger_wav']).is_file())

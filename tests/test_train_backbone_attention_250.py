import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

from fireredtts3_local import train_backbone_attention_250 as job
from test_training_batch import tiny_core


def prepare_pilots(source):
    reference = source/'voice.wav'
    sf.write(reference, np.zeros(800), 16000)
    specs = [[(text, s) for text in ('question', 'cities', 'timofey') for s in (1234, 1235)],
             [('cities', s) for s in range(1236, 1240)]]
    for folder, controls in zip(job.PILOTS, specs, strict=True):
        pilot = source/folder
        pilot.mkdir()
        cases = [dict(id=f'{text}_seed{seed}', text=text, seed=seed, reference_audio=str(reference),
                      reference_text='Голос.') for text, seed in controls]
        benchmark = pilot/'benchmark.jsonl'
        benchmark.write_text('\n'.join(map(json.dumps, cases)))
        _, digest = job.load_benchmark(benchmark)
        baseline = [dict(c, asr={'wer': 0}, voice_similarity=1, final_voice_similarity=1,
                         end_minus_start_rms_db=0) for c in cases]
        job.write_json(pilot/'report.json', dict(complete=True, settings=job.listening.SETTINGS,
            recipes={'lora': job.vox_recipe_transfer.transfer_recipe('lora', 'data')},
            benchmark_sha256=digest, results={'base': baseline}))


class BackboneRunTests(unittest.TestCase):
    def test_joint_run_resumes_after_listening_and_uses_only_backbone_for_both_packs(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)
            prepare_pilots(source)
            output = source/job.JOINT_RUN
            calls = []
            saved_plans = []

            def train(dataset, out, mode, until, *, resume, **kwargs):
                self.assertEqual((out, mode), (output, 'lora'))
                self.assertEqual(kwargs, dict(fork_from=None, shared_features=source/'features',
                    backbone_only=False, recovery=job.vox_recovery.FOUR_PATCH,
                    checkpoint_steps=(), expected_dataset_sha256='data'))
                self.assertEqual(resume, until == 500)
                if until == 500:
                    report = json.loads((out/'audio/step-0250/backbone-complete/report.json').read_text())
                    self.assertTrue(report['audio_complete'])
                    self.assertEqual(len(report['results']), 58)
                calls.append(until)
                checkpoint = out/'lora'/f'step-{until:04d}'
                checkpoint.mkdir(parents=True)
                recipe = dict(job.vox_recipe_transfer.transfer_recipe('lora', 'data'),
                              acoustic_recovery=job.vox_recovery.FOUR_PATCH, recovery_start_step=0)
                job.write_json(checkpoint/'checkpoint.json', dict(step=until, recipe=recipe))
                (checkpoint/'adapter_config.json').write_text('{}')
                (checkpoint/'adapter_model.safetensors').write_bytes(str(until).encode())

            generated = (np.ones(2400, dtype=np.float32)*.1, 24000, {'finish_reason': 'stop'})
            with (patch.object(job.vox_recipe_transfer, 'train', side_effect=train),
                  patch.object(job.listening, 'ModelManager') as manager,
                  patch.object(job.listening, 'recognize'),
                  patch.object(job.listening, 'generate_case', side_effect=[generated, InterruptedError]) as gen):
                manager.return_value.load_base.return_value.tts_core = job.vox_recipe_transfer.configure(tiny_core(), 'lora')
                with self.assertRaises(InterruptedError):
                    job.run(source, source/'data', joint_to_500=True)
                self.assertEqual(calls, [250])
                gen.side_effect = None
                gen.return_value = generated
                job.run(source, source/'data', joint_to_500=True)
                job.run(source, source/'data', joint_to_500=True)
                self.assertEqual(calls, [250, 500])
                self.assertEqual(gen.call_count, 117)  # 116 WAVs and one interrupted generation
            replay = json.loads((output/'replay.json').read_text())
            self.assertEqual(replay['checkpoint_steps'], [250, 500])
            self.assertEqual(replay['audio_steps'], [250, 500])
            for step in calls:
                info = json.loads((output/'audio'/f'step-{step:04d}/backbone-complete/report.json').read_text())
                plan = info['plan']
                self.assertEqual(list(plan['checkpoints']), [f'step-{step:04d}'])
                self.assertEqual(plan['variants'], ['backbone'])
                self.assertFalse(plan['include_base'])
                self.assertTrue(all(r['active_layers'] == {'backbone': 8, 'dit': 0} for r in info['results'].values()))
                saved_plans.append(plan['cases'])
            self.assertEqual(saved_plans[0], saved_plans[1])

    def test_stronger_resonant_starts_from_base_and_renders_only_after_500(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)
            prepare_pilots(source)
            output = source/job.STRONG_RUN
            profile = dict(job.vox_recovery.RESONANT, weight=.30)
            # Existing clean checkpoints, including 500, must not be resumed.
            for name, step in [(job.RUN, 250), (job.CONTINUATION, 500)]:
                old = source/name/'lora'/f'step-{step:04d}'
                old.mkdir(parents=True)
                (old/'checkpoint.json').write_text('{}')

            def train(dataset, out, mode, until, *, resume, **kwargs):
                self.assertEqual((out, mode, until), (output, 'lora', 500))
                self.assertEqual(kwargs, dict(fork_from=None, shared_features=source/'features', recovery=profile,
                    backbone_only=True, checkpoint_steps=(100, 250), expected_dataset_sha256='data'))
                replay = json.loads((out/'replay.json').read_text())
                self.assertEqual(replay['recipe']['acoustic_recovery'], profile)
                self.assertEqual(replay['recipe']['recovery_start_step'], 0)
                self.assertNotIn('rise_gate', replay['recipe'])
                self.assertEqual(replay['audio_steps'], [500])
                self.assertNotIn('parent_checkpoint', replay)
                self.assertEqual(resume, (out/'lora/step-0250/checkpoint.json').exists())
                step = 500 if resume else 250
                checkpoint = out/'lora'/f'step-{step:04d}'
                checkpoint.mkdir(parents=True)
                job.write_json(checkpoint/'checkpoint.json', dict(step=step, recipe=replay['recipe']))
                if step == 250:
                    raise InterruptedError
                (checkpoint/'adapter_config.json').write_text('{}')
                (checkpoint/'adapter_model.safetensors').write_bytes(b'weights')

            with (patch.object(job.vox_recipe_transfer, 'train', side_effect=train) as trainer,
                  patch.object(job.listening, 'render') as render):
                with self.assertRaises(ValueError):
                    job.run(source, source/'data', continue_to_500=True, stronger_recovery_loss=True)
                trainer.assert_not_called()
                with self.assertRaises(InterruptedError):
                    job.run(source, source/'data', stronger_recovery_loss=True)
                render.assert_not_called()
                job.run(source, source/'data', stronger_recovery_loss=True)
                render.assert_called_once()
                path, plan = render.call_args.args
                self.assertEqual(path, output/'audio/step-0500/backbone-complete')
                self.assertEqual(list(plan['checkpoints']), ['step-0500'])
                self.assertEqual(plan['variants'], ['backbone'])
                self.assertFalse(plan['include_base'])
                self.assertEqual(len(plan['cases']), 58)

    def test_interrupted_training_has_no_audio_then_renders_only_final_suite(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)
            dataset = source/'data'
            run = source/job.RUN
            prepare_pilots(source)

            def train(data, out, mode, until, *, resume, **kwargs):
                self.assertIn(until, (250, 500))
                self.assertEqual((data, out, mode), (dataset, run if until == 250 else source/job.CONTINUATION, 'lora'))
                parent = run/'lora/step-0250' if until == 500 and not (out/'lora/step-0500').exists() else None
                self.assertEqual(kwargs, dict(shared_features=source/'features', backbone_only=True,
                    recovery=job.vox_recovery.FOUR_PATCH, checkpoint_steps=(100,) if until == 250 else (),
                    expected_dataset_sha256='data', fork_from=parent))
                if until == 500:
                    self.assertTrue(resume)
                replay = json.loads((out/'replay.json').read_text())
                self.assertEqual(replay['audio_steps'], [until])
                step = until if resume else 100
                checkpoint = out/'lora'/f'step-{step:04d}'
                if checkpoint.exists():
                    return
                checkpoint.mkdir(parents=True)
                job.write_json(checkpoint/'checkpoint.json', dict(step=step, recipe=replay['recipe']))
                if step == 100:
                    raise InterruptedError
                (checkpoint/'adapter_config.json').write_text('{}')
                (checkpoint/'adapter_model.safetensors').write_bytes(b'weights')
                (checkpoint/'training_state.pt').write_bytes(b'optimizer, scheduler, RNG')
                (checkpoint/'firered_lora_metadata.json').write_text('{}')

            generated = (np.ones(2400, dtype=np.float32)*.1, 24000, {'finish_reason': 'stop'})
            with (patch.object(job.vox_recipe_transfer, 'train', side_effect=train),
                  patch.object(job.listening, 'ModelManager') as manager,
                  patch.object(job.listening, 'recognize'),
                  patch.object(job.listening, 'generate_case', return_value=generated) as gen):
                manager.return_value.load_base.return_value.tts_core = job.vox_recipe_transfer.configure(
                    tiny_core(), 'lora', backbone_only=True)
                with self.assertRaises(InterruptedError):
                    job.run(source, dataset)
                self.assertFalse((run/'audio').exists())
                with self.assertRaises(FileNotFoundError):
                    job.run(source, dataset, continue_to_500=True)
                self.assertFalse((source/job.CONTINUATION/'replay.json').exists())
                job.run(source, dataset)
                job.run(source, dataset)
                self.assertEqual(gen.call_count, 58)
                pack = run/'audio/step-0250/backbone-complete'
                self.assertEqual(len(list(run.rglob('*.wav'))), 58)
                report = json.loads((pack/'report.json').read_text())
                self.assertEqual(report['plan']['adapter_profile'], job.BACKBONE_ATTENTION)
                self.assertTrue(report['audio_complete'])
                self.assertTrue(all(row['active_layers'] == {'backbone': 8, 'dit': 0}
                                    for row in report['results'].values()))
                job.run(source, dataset, continue_to_500=True)
                job.run(source, dataset, continue_to_500=True)
                self.assertEqual(gen.call_count, 116)
                new = source/job.CONTINUATION
                self.assertEqual([p.name for p in (new/'audio').iterdir()], ['step-0500'])
                final = json.loads((new/'audio/step-0500/backbone-complete/report.json').read_text())
                self.assertEqual(final['plan']['cases'], report['plan']['cases'])
                self.assertEqual(final['plan']['variants'], ['backbone'])
                self.assertEqual(json.loads((pack/'report.json').read_text()), report)
                (run/'lora/step-0250/training_state.pt').write_bytes(b'different state')
                with self.assertRaisesRegex(ValueError, 'different training or listening inputs'):
                    job.run(source, dataset, continue_to_500=True)

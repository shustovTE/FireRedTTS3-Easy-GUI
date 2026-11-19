import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

from fireredtts3_local import train_backbone_attention_250 as job
from test_training_batch import tiny_core


class BackboneRunTests(unittest.TestCase):
    def test_interrupted_training_has_no_audio_then_renders_only_final_suite(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)
            dataset = source/'data'
            run = source/job.RUN
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

            def train(data, out, mode, until, *, resume, **kwargs):
                self.assertEqual((data, out, mode, until), (dataset, run, 'lora', 250))
                self.assertEqual(kwargs, dict(shared_features=source/'features', backbone_only=True,
                    recovery=job.vox_recovery.FOUR_PATCH, checkpoint_steps=(100,), expected_dataset_sha256='data'))
                replay = json.loads((run/'replay.json').read_text())
                self.assertEqual(replay['audio_steps'], [250])
                step = 250 if resume else 100
                checkpoint = run/'lora'/f'step-{step:04d}'
                if checkpoint.exists():
                    return
                checkpoint.mkdir(parents=True)
                job.write_json(checkpoint/'checkpoint.json', dict(step=step, recipe=replay['recipe']))
                if step == 100:
                    raise InterruptedError
                (checkpoint/'adapter_config.json').write_text('{}')
                (checkpoint/'adapter_model.safetensors').write_bytes(b'weights')

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

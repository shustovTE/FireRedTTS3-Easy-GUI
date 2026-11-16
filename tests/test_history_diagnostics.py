"""Real tiny upstream core: causal interventions, native parity and codec controls."""
import copy
from dataclasses import asdict
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

from fireredtts3_local import cfg_diagnostics as shared
from fireredtts3_local import history_diagnostics as diag
from fireredtts3_local import history_probe as probe
from fireredtts3_local.training_data import dataset_fingerprint, file_digest, load_manifest
from fireredtts3_local.training_run import TrainingOptions, prepare_run_data
from test_training_batch import tiny_core
from test_hybrid_training import hybrid
import test_cfg_diagnostics as fixtures


class Codec(torch.nn.Module):
    sample_rate = 1000
    downsample_rate = 80

    @staticmethod
    def pad_to_multiple_of(audio, multiple):
        return F.pad(audio, ((-audio.shape[-1]) % multiple, 0))

    def encode(self, audio, rate):
        assert rate == self.sample_rate
        return audio.reshape(1, -1, 80).mean(-1, keepdim=True).repeat(1, 1, 4)

    def decode(self, latents):
        return latents[:, :, 0].repeat_interleave(80, dim=1), self.sample_rate


class Speaker(torch.nn.Module):
    def forward(self, audio, rate):
        return torch.ones(1, 6)


class Pipeline:
    def __init__(self, core):
        self.tts_core = core
        self.redae = Codec()
        self.spk_extractor = Speaker()
        self.device = torch.device('cpu')

    def _apply_frontend(self, *, text, language, do_split):
        assert not do_split
        return text, language, [text]

    def _tokenize_text(self, text):
        return torch.tensor([[1, 2, 3]])


class Manager(fixtures.FakeManager):
    def load_base(self, **kwargs):
        assert self.base is None
        self.base = Pipeline(copy.deepcopy(self.original).eval())
        self.loads += 1
        return self.base


class HistoryMathTests(unittest.TestCase):
    def setUp(self):
        threads = torch.get_num_threads(); torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, threads)
        torch.manual_seed(322)
        self.core = tiny_core().eval()
        self.inputs = dict(tokens=torch.tensor([[1, 2, 3]]), prompt=torch.randn(1, 4, 4),
                           target=torch.randn(1, 24, 4), speaker=torch.ones(1, 6))
        self.settings = dict(shared.SETTINGS)

    def test_self_history_matches_native_for_base_and_hybrid_and_preserves_rng(self):
        adapted = hybrid(copy.deepcopy(self.core)).eval()
        with torch.no_grad():
            for p in adapted.parameters():
                if p.requires_grad: p.add_(torch.randn_like(p)*.002)
        for core in (self.core, adapted):
            for prompt_patches in (1, 9):
                inputs = dict(self.inputs, prompt=torch.randn(1, 2*prompt_patches, 4))
                before = {k: v.clone() for k, v in core.state_dict().items()}
                rng = torch.get_rng_state()
                free, free_stops = probe.native_free(core, inputs, 1234, self.settings)
                own, own_stops = probe.fixed_history(core, inputs, teacher=False, seed=1234, settings=self.settings)
                count = min(free.shape[1], own.shape[1])
                torch.testing.assert_close(own[:, :count], free[:, :count], rtol=0, atol=0)
                self.assertEqual(own_stops[:min(len(own_stops), len(free_stops))],
                                 free_stops[:min(len(own_stops), len(free_stops))])
                torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
                for key, value in core.state_dict().items():
                    torch.testing.assert_close(value, before[key], rtol=0, atol=0)

    def test_teacher_does_not_use_current_or_future_target_and_predictions_are_not_copies(self):
        result, _ = probe.fixed_history(self.core, self.inputs, teacher=True, seed=9, settings=self.settings)
        changed = dict(self.inputs, target=self.inputs['target'].clone())
        changed['target'][:, 8:] += 7
        after, _ = probe.fixed_history(self.core, changed, teacher=True, seed=9, settings=self.settings)
        # Changing target patch j cannot affect prediction j, only j+1 onward.
        torch.testing.assert_close(result[:, :10], after[:, :10], rtol=0, atol=0)
        self.assertFalse(torch.allclose(result[:, 10:], after[:, 10:]))
        self.assertFalse(torch.allclose(result, self.inputs['target']))
        own, _ = probe.fixed_history(self.core, self.inputs, teacher=False, seed=9, settings=self.settings)
        other_own, _ = probe.fixed_history(self.core, changed, teacher=False, seed=9, settings=self.settings)
        torch.testing.assert_close(own, other_own, rtol=0, atol=0)
        torch.testing.assert_close(own[:, :2], result[:, :2], rtol=0, atol=0)

    def test_fixed_horizons_ignore_stop_but_native_control_respects_it(self):
        with torch.no_grad():
            self.core.stop_head.weight.zero_(); self.core.stop_head.bias.fill_(10)
        free, free_stops = probe.native_free(self.core, self.inputs, 5, self.settings)
        own, own_stops = probe.fixed_history(self.core, self.inputs, teacher=False, seed=5, settings=self.settings)
        teacher, stops = probe.fixed_history(self.core, self.inputs, teacher=True, seed=5, settings=self.settings)
        self.assertEqual(free.shape[1], 6*self.core.patch_size)
        self.assertEqual(own.shape, self.inputs['target'].shape)
        self.assertEqual(teacher.shape, own.shape)
        self.assertEqual(len(free_stops), 7)
        self.assertEqual(len(stops), 13)
        self.assertEqual(len(own_stops), 13)

    def test_incorrect_history_loop_fails_native_control_before_decoding(self):
        original = probe.fixed_history
        def corrupt(*args, **kwargs):
            result, stops = original(*args, **kwargs)
            return (result if kwargs['teacher'] else result+10), stops
        with patch.object(probe, 'fixed_history', side_effect=corrupt):
            with self.assertRaisesRegex(ValueError, 'control mismatch'):
                probe.history_check(Pipeline(self.core), self.inputs, 1234, self.settings)

    def test_codec_padding_crops_left_and_keeps_reference_context(self):
        class IdentityCodec(Codec):
            downsample_rate = 1
            def decode(self, latents):
                return latents[:, :, 0], self.sample_rate
        tts = Pipeline(self.core); tts.redae = IdentityCodec()
        source = torch.tensor([[.2, .4, .6]])
        inputs = dict(source=source, prompt=torch.ones(1, 2, 4),
                      target=torch.tensor([[[0.]*4, [.2]*4, [.4]*4, [.6]*4]]),
                      target_pad=1, prompt_samples=2, source_samples=3, rate=1000)
        results = probe.codec_check(tts, inputs)
        for wave in results.values():
            np.testing.assert_array_equal(wave, source[0].numpy())
        np.testing.assert_array_equal(probe.decode_target(tts, inputs, inputs['target'], fixed=False),
                                      np.array([0., .2, .4, .6], dtype=np.float32))

    def test_codec_gain_metric_does_not_confuse_natural_level_growth_with_distortion(self):
        rate = 1000
        t = np.arange(9000)/rate
        source = np.sin(t*180)*np.linspace(.05, .5, len(t))
        same = probe.reconstruction_metrics(source, source, rate)
        constant = probe.reconstruction_metrics(source, source*2, rate)
        drifted = probe.reconstruction_metrics(source, source*np.linspace(1, 3, len(t)), rate)
        self.assertGreater(probe.levels(source, rate)['end_minus_start_rms_db'], 5)
        self.assertEqual(same['tail_gain_minus_head_gain_db'], 0)
        self.assertEqual(same['spectral_convergence'], 0)
        self.assertAlmostEqual(constant['tail_gain_minus_head_gain_db'], 0)
        self.assertGreater(drifted['tail_gain_minus_head_gain_db'], 4)
        short = probe.levels(source[:1500], rate)
        self.assertEqual(short['rms_window_seconds'], .5)
        self.assertGreater(short['end_minus_start_rms_db'], 0)


class HistoryJobTests(unittest.TestCase):
    def setUp(self):
        self.f = fixtures.DiagnosticTests(); self.addCleanup(self.f.doCleanups); self.f.setUp()
        f = self.f
        manifests = {}
        for group_index, (group, speakers) in enumerate([('train', ['train']), ('dialogs', ['dialogs']), ('cv', ['cv-a', 'cv-b'])]):
            rows = []
            for s, speaker in enumerate(speakers):
                for i in range(2):
                    path = f.root/f'{speaker}-{i}.wav'
                    length = 2017+i*311+s*97
                    sf.write(path, np.sin(np.arange(length)*(.013+i*.001+s*.003+group_index*.01))*.2, 1000)
                    rows.append(dict(audio=str(path), text=f'Реплика {group} диктора {speaker} номер {i}.',
                                     speaker_id=speaker, language='Russian', source_id=f'{speaker}-{i}'))
            manifest = f.root/f'{group}.jsonl'
            manifest.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            manifests[group] = manifest
        f.recipe['seed'] = 1234
        f.recipe['options'] = asdict(TrainingOptions.parse({**f.recipe['options'],
            'validation_manifest': str(manifests['dialogs']), 'adaptation_validation_manifest': str(manifests['cv']),
            'validation_scope': 'recording', 'max_target_seconds': 30., 'max_prompt_seconds': 30., 'replay_ratio': 0.}))
        train, dialogs, _, cv = prepare_run_data(manifests['train'], f.recipe['options'], 1234,
                                                 include_adaptation_validation=True)
        for key, pairs in [('train_fingerprint', train), ('validation_fingerprint', dialogs),
                           ('adaptation_validation_fingerprint', cv)]:
            f.recipe[key] = dataset_fingerprint(pairs.rows)
        (f.run/'recipe.json').write_text(json.dumps(f.recipe))
        for path in f.run.glob('*/firered_lora_metadata.json'):
            value = json.loads(path.read_text()); value['recipe'] = f.recipe; path.write_text(json.dumps(value))
        self.data_recipe = f.root/'data-recipe.json'
        self.data_recipe.write_text(json.dumps(dict(manifest_path=str(manifests['train']))))
        self.manifests = manifests

    def plan(self):
        return diag.make_plan(self.f.run, self.f.output, seeds=[1234], min_long_seconds=2.,
                              data_recipe=self.data_recipe)

    def test_plan_selects_heldout_real_records_and_rejects_changed_data(self):
        plan = self.plan()
        self.assertEqual(plan['total_cases'], 27)
        self.assertEqual(len(plan['cases']), 3)
        self.assertEqual(len({c['target']['speaker_id'] for c in plan['cases'][:2]}), 2)
        for case in plan['cases']:
            self.assertEqual(case['target']['speaker_id'], case['reference']['speaker_id'])
            self.assertNotEqual(case['target']['audio_sha256'], case['reference']['audio_sha256'])
        diag.verify_plan(plan)
        shared.verify_plan(plan)
        with self.assertRaisesRegex(ValueError, 'Need 2 real'):
            diag.make_plan(self.f.run, self.f.output, min_long_seconds=15., data_recipe=self.data_recipe)
        lines = self.manifests['cv'].read_text().splitlines()
        row = json.loads(lines[0]); row['text'] += ' Изменено.'; lines[0] = json.dumps(row)
        self.manifests['cv'].write_text('\n'.join(lines)+'\n')
        with self.assertRaisesRegex(ValueError, 'data changed'):
            diag.verify_plan(plan)

    def test_complete_native_pipeline_resume_common_gain_and_asr_archives(self):
        plan = self.plan()
        hashes = {p: file_digest(p) for p in self.f.run.rglob('*') if p.is_file()}
        generated = diag.execute(plan, manager=Manager(self.f.original), log=lambda _: None)
        self.assertTrue(generated['complete'])
        self.assertEqual(len(generated['results']), 27)
        for case in plan['cases']:
            rows = [r for r in generated['results'] if r['id']==case['id']]
            self.assertEqual(len({r['pcm_gain'] for r in rows}), 1)
            source = next(r for r in rows if r['mode']=='source')
            for row in rows:
                if row['mode'] != 'free': self.assertEqual(row['seconds'], source['seconds'])
                if row['mode'] in ('teacher', 'self_fixed', 'free'):
                    self.assertTrue(row['controls']['native_common_prefix']['passed'])
                    self.assertTrue(row['controls']['first_patch']['passed'])
        files = {name: file_digest(self.f.output/name) for row in generated['results'] for name in row['artifacts']}
        manager = Manager(self.f.original)
        resumed = diag.execute(plan, resume=True, manager=manager, log=lambda _: None)
        self.assertEqual(manager.loads, 0)
        self.assertEqual(generated, resumed)
        calls = []
        def transcribe(path, **settings):
            calls.append(path)
            self.assertIsInstance(path, np.ndarray)
            self.assertTrue(np.isfinite(path).all())
            self.assertLessEqual(float(np.abs(path).max()), .980001)
            if len(calls)==3: raise RuntimeError('ASR interrupted')
            self.assertEqual(settings, plan['asr_settings'])
            return iter([SimpleNamespace(start=0., end=1., text='Проверка.', words=[])]), None
        recognizer = SimpleNamespace(transcribe=transcribe)
        with patch.object(diag, 'load_asr', return_value=(recognizer, {'identity':'test'})):
            with self.assertRaisesRegex(RuntimeError, 'ASR interrupted'):
                diag.score(plan, log=lambda _: None)
            report = diag.score(plan, log=lambda _: None)
        self.assertTrue(report['complete'])
        self.assertEqual(len(calls), 28)  # completed rows skipped; failed third row retried
        diag.execute(plan, resume=True, manager=Manager(self.f.original), log=lambda _: None)
        with patch.object(diag, 'load_asr', return_value=(recognizer, {'identity':'test'})):
            diag.score(plan, log=lambda _: None)
        self.assertEqual(len(calls), 28)
        self.assertNotIn('pending', (self.f.output/'summary.md').read_text())
        with zipfile.ZipFile(self.f.output/'diagnostics-report.zip') as z:
            self.assertIn('asr_report.json', z.namelist())
            self.assertNotIn('plan.json', z.namelist())
        names = []
        for path in self.f.output.glob('diagnostics-audio-*.zip'):
            with zipfile.ZipFile(path) as z: names += z.namelist()
        self.assertEqual(len(names), 27)
        self.assertTrue(all('.float.wav' not in p and Path(p).name!='reference.wav' for p in names))
        for name, digest in files.items(): self.assertEqual(file_digest(self.f.output/name), digest)
        for path, digest in hashes.items(): self.assertEqual(file_digest(path), digest)
        (self.f.output/generated['results'][0]['audio']).write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'audio changed'):
            diag.score(plan, log=lambda _: None)

    def test_asr_uses_float_audio_independently_of_listening_gain(self):
        root = self.f.root
        wave = np.sin(np.arange(16000)*.02).astype(np.float32)*.2
        sf.write(root/'raw.wav', wave, 16000, subtype='FLOAT')
        sf.write(root/'quiet.wav', wave*.000001, 16000, subtype='PCM_16')
        result, gain = diag.asr_input(root, dict(float_audio='raw.wav',audio='quiet.wav',pcm_gain=.000001))
        np.testing.assert_array_equal(result, wave)
        self.assertEqual(gain, 1.)

    def test_interrupted_generation_reuses_complete_groups(self):
        plan = self.plan()
        original = diag.history_check
        calls = []
        def interrupted(*args, **kwargs):
            calls.append(1)
            if len(calls)==2: raise RuntimeError('synthetic interruption')
            return original(*args, **kwargs)
        with patch.object(diag, 'history_check', side_effect=interrupted):
            with self.assertRaisesRegex(RuntimeError, 'synthetic interruption'):
                diag.execute(plan, manager=Manager(self.f.original), log=lambda _: None)
        partial = json.loads((self.f.output/'report.json').read_text())
        saved = {p: file_digest(self.f.output/p) for r in partial['results'] for p in r['artifacts']}
        with patch.object(diag, 'history_check', wraps=original) as check:
            report = diag.execute(plan, resume=True, manager=Manager(self.f.original), log=lambda _: None)
        self.assertTrue(report['complete'])
        self.assertEqual(check.call_count, 5)
        for name, digest in saved.items(): self.assertEqual(file_digest(self.f.output/name), digest)

    def test_worker_dispatches_both_history_phases(self):
        job = self.f.root/'job'; job.mkdir()
        (job/'status.json').write_text(json.dumps(dict(cwd=str(Path.cwd()),resume=True)))
        (job/'plan.json').write_text(json.dumps(self.plan()))
        fd = os.open(self.f.root/'lock', os.O_CREAT | os.O_WRONLY, 0o600)
        with patch('sys.argv', ['cfg_diagnostics', '_worker', str(job), str(fd)]), \
                patch.object(shared, 'run_commands', return_value=4) as run:
            self.assertEqual(shared.main(), 4)
        commands = run.call_args.args[2]
        self.assertEqual([p for p,_ in commands], ['diagnostics', 'asr'])
        self.assertTrue(all(c[3]=='fireredtts3_local.history_diagnostics' for _,c in commands))
        self.assertIn('--resume', commands[0][1])


if __name__ == '__main__':
    unittest.main()

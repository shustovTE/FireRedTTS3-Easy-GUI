import csv
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf
import torch

from fireredtts3_local import feedback_probe as probe
from test_training_batch import tiny_core


class FeedbackProbeTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(29)
        self.core = tiny_core().eval()
        self.inputs = dict(spk_emb=torch.randn(1, 6), text_tokens=torch.tensor([[1, 2, 3]]),
                           prompt_latents=torch.randn(1, 6, 4), n_timesteps=2,
                           stop_threshold=1.0, max_gen_steps=9)

    def generate(self):
        torch.manual_seed(63)
        return self.core.generate(**self.inputs)

    def test_exact_native_control_and_independent_routes(self):
        clean = self.generate()
        anchor = 3
        with probe.observe_feedback(self.core, anchor) as control:
            self.assertTrue(torch.equal(self.generate(), clean))
        state = torch.get_rng_state()
        delta = probe.make_delta(control.patches[anchor], .02, 42)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        self.assertAlmostEqual(float(delta.norm()/control.patches[anchor].norm()), .02, places=6)
        for route in ('sham', *probe.ROUTES):
            with self.subTest(route=route):
                with probe.observe_feedback(self.core, anchor, route, control=control, delta=delta) as branch:
                    actual = self.generate()
                end = self.inputs['prompt_latents'].shape[1]+(anchor+1)*self.core.patch_size
                self.assertTrue(torch.equal(actual[:, :end], clean[:, :end]))
                self.assertEqual(branch.dit_reads, [4,5] if route in ('dit_history','both_histories') else [])
                self.assertEqual(branch.backbone_reads, [3] if route in ('backbone_history','both_histories') else [])
                self.assertEqual(torch.equal(actual, clean), route == 'sham')
                rows = probe.paired_latents(control, branch, anchor, delta, .16)
                self.assertEqual([r['hop'] for r in rows], [1, 2, 3, 4, 5])
                self.assertEqual(rows[0]['gain'] > 0, route != 'sham')
        self.assertTrue(torch.equal(self.generate(), clean))

    def test_sliding_history_changes_only_the_anchor_slot(self):
        # Check the exact frames passed as direct DiT history.
        original = self.core._flow_one_step
        inputs = []
        def zero_history_effect(**kwargs):
            inputs.append(kwargs['hist_latents'].clone())
            kwargs['hist_latents'] = torch.zeros_like(kwargs['hist_latents'])
            return original(**kwargs)
        # Hold model output invariant to direct history to distinguish injected frames
        # from subsequent, naturally diverging predictions.
        with patch.object(self.core, '_flow_one_step', zero_history_effect):
            with probe.observe_feedback(self.core, 2) as control:
                clean = self.generate()
            saved_inputs, inputs = inputs, []
            delta = probe.make_delta(control.patches[2], .02, 7)
            with probe.observe_feedback(self.core, 2, 'dit_history', control=control, delta=delta):
                self.assertTrue(torch.equal(clean, self.generate()))
        p = self.core.patch_size
        for i, hist in enumerate(saved_inputs):
            expected = hist.clone()
            if i in (3,4):
                start = p if i == 3 else 0
                expected[:, start:start+p] += delta
            self.assertTrue(torch.equal(inputs[i], expected))

    def test_rng_guard_and_cleanup_on_failure(self):
        with probe.observe_feedback(self.core, 2) as control:
            self.generate()
        control.rng[0] = torch.zeros_like(control.rng[0])
        with self.assertRaisesRegex(RuntimeError, 'noise stream'):
            with probe.observe_feedback(self.core, 2, 'sham', control=control):
                self.generate()
        with probe.observe_feedback(self.core, 2):
            self.generate()

    def test_end_to_end_reports_and_short_case(self):
        rate = 24000
        # Replace the expensive codec only. All branches still use real native autoregression.
        tts = SimpleNamespace(tts_core=SimpleNamespace(get_base_model=lambda: self.core),
                              redae=SimpleNamespace(sample_rate=rate, downsample_rate=480))
        def render(*args):
            out = self.generate()[:, self.inputs['prompt_latents'].shape[1]:]
            audio = out.mean(-1)[0].repeat_interleave(480).numpy()
            return audio, rate, dict(sample_rate=rate, generated_patches=9,
                                     finish_reason='max_patches', stop_scores=[0.]*9)
        with TemporaryDirectory() as temp, patch.object(probe, 'generate_case', render):
            root = Path(temp)
            source = root/'source.wav'
            sf.write(source, render()[0], rate, subtype='FLOAT')
            case = dict(id='cities', seed=42)
            summary = probe.replay_case(tts, case, {}, source, root, 2, .02)
            self.assertTrue(summary['archive_pcm_exact'] and summary['sham_exact'])
            self.assertEqual(summary['status'], 'tested')
            for mode in probe.ROUTES:
                self.assertEqual(summary['branches'][mode]['compared_hops'], 6)
                self.assertTrue((root/f'{mode}.wav').is_file())
            self.assertEqual(len((root/'patches.csv').read_text().splitlines()), 1+6*3)
            self.assertEqual(probe.replay_case(tts, case, {}, source, root, 12, .02)['status'],
                             'insufficient_continuation')

    def test_windows_stop_at_shorter_wave(self):
        t = np.arange(60000)/24000
        wav = np.sin(2*np.pi*2220*t)
        rows = probe.paired_windows(wav, 2*wav[:30000], 24000)
        self.assertEqual([r['end_s'] for r in rows], [1., 1.25])
        for row in rows:
            self.assertAlmostEqual(row['rms_delta_db'], 6.0206, places=4)
            self.assertAlmostEqual(row['resonance_delta_db'], 0, places=6)

    def test_paired_command_opens_both_profiles_matches_cases_and_resumes(self):
        rate = 24000
        tts = SimpleNamespace(tts_core=SimpleNamespace(get_base_model=lambda: self.core, patch_size=self.core.patch_size),
                              redae=SimpleNamespace(sample_rate=rate, downsample_rate=960))
        def render(*args):
            out = self.generate()[:, self.inputs['prompt_latents'].shape[1]:]
            audio = out.mean(-1)[0].repeat_interleave(960).numpy()
            return audio, rate, dict(sample_rate=rate, generated_patches=9,
                                     finish_reason='max_patches', stop_scores=[0.]*9)
        with TemporaryDirectory() as temp:
            root = Path(temp)
            ref = root/'reference.wav'
            sf.write(ref, np.zeros(rate), rate)
            case = dict(id='cities_seed1235', filename='controls/cities_seed1235.wav', seed=1235,
                        text='Проверка.', language='Russian', reference_text='Голос.',
                        reference_audio=str(ref), reference_sha256=probe.file_digest(ref))
            runs = [root/'joint-run', root/'backbone-run']
            for path, profile in zip(runs, ('vox_attention', probe.BACKBONE_ATTENTION)):
                ckpt = path/'lora'/'step-0500'
                ckpt.mkdir(parents=True)
                (ckpt/'adapter_model.safetensors').write_bytes(b'fixture')
                (ckpt/'adapter_config.json').write_text('{}')
                probe.write_json(ckpt/'checkpoint.json', dict(step=500, recipe={'adapter_profile': profile}))
                pack = path/'audio'/'step-0500'/'backbone-complete'
                wav = pack/case['filename']
                wav.parent.mkdir(parents=True)
                sf.write(wav, render()[0], rate, subtype='FLOAT')
                plan = dict(adapter_profile=profile, variants=['backbone'], attention='sdpa',
                            settings=probe.SETTINGS, cases=[case], base_sha256=probe.BASE_SHA,
                            checkpoints=probe.checkpoint_files(ckpt))
                saved = dict(case=case['id'], active_layers={'backbone':8,'dit':0}, wav_sha256=probe.file_digest(wav))
                probe.write_json(pack/'report.json', dict(audio_complete=True, plan=plan, results={case['filename']:saved}))
            args = dict(checkpoint=runs[0]/'lora'/'step-0500', compare_run=runs[1], output=root/'out',
                        cases=[case['id']], at_seconds=.08, fraction=.02)
            with (patch.object(probe, 'model_identity', return_value=probe.BASE_SHA),
                  patch.object(probe, 'ModelManager') as manager, patch.object(probe, 'route') as route,
                  patch.object(probe, 'generate_case', side_effect=render) as gen):
                manager.return_value.load_base.return_value = tts
                probe.run(**args)
                self.assertEqual([c.kwargs['backbone_only'] for c in route.call_args_list], [False, True])
                self.assertEqual(gen.call_count, 10)
                with (args['output']/'comparison.csv').open() as f:
                    rows = list(csv.DictReader(f))
                self.assertEqual([r['route'] for r in rows], list(probe.ROUTES))
                for row in rows:
                    self.assertEqual(row['common_hops'], '7')
                    self.assertEqual(row['status'], 'tested')
                probe.run(**args)
                self.assertEqual(gen.call_count, 10)
                path = runs[1]/'audio'/'step-0500'/'backbone-complete'/'report.json'
                report = probe.read_json(path)
                report['plan']['cases'][0]['reference_text'] = 'Другой текст.'
                probe.write_json(path, report)
                with self.assertRaisesRegex(ValueError, 'text, reference'):
                    probe.run(**args)

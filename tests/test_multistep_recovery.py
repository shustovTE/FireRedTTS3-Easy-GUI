import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf
import torch

from fireredtts3_local import vox_recipe_sweep as sweep, vox_recovery as recovery, vox_recovery_probe as probe
from fireredtts3_local.training_batch import Example, set_training
from test_training_batch import tiny_core


class MultistepTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(1)

    def test_multistep_supervises_all_real_targets_and_conditions_on_detached_predictions(self):
        torch.manual_seed(1)
        core = tiny_core()
        set_training(core)
        with torch.no_grad():
            core.stop_head.weight.zero_()
            core.stop_head.bias.fill_(-12.)
        # Downsample by two; four signal channels per frame for the downsized core.
        codec = SimpleNamespace(sample_rate=1000, downsample_rate=2,
                                encode=lambda w, _: w[:, ::2, None].expand(-1, -1, 4))
        tts = SimpleNamespace(tts_core=core, redae=codec)
        with tempfile.TemporaryDirectory() as tmp:
            src = str(Path(tmp)/'src.wav')
            wave = torch.sin(torch.arange(80)*.42)[None]
            sf.write(src, wave[0], 1000, subtype='FLOAT')
            z = codec.encode(wave, 1000)
            ex = Example(torch.tensor([1,2]), z[:, :4], z, torch.zeros(1,6))
            # Both examples use the same real source; one has only two clean future patches.
            seen = []
            def fake_patch(model, c, s, h, options):
                self.assertFalse(torch.is_grad_enabled())
                self.assertFalse(model.training)
                self.assertEqual(options.n_timesteps, 10)
                seen.append(h.clone())
                return torch.full((len(h), core.patch_size, 4), 3. + len(seen))
            with patch('random.Random.randint', side_effect=[10,18]), \
                    patch.object(recovery, 'sample_native_patches', fake_patch), \
                    patch.object(recovery, 'predict', wraps=recovery.predict) as predict:
                counts = [0]*4
                loss, accepted = recovery.recovery_loss(tts, [ex, ex], [{'audio':src}]*2, 3,
                                                        config=recovery.FOUR_PATCH, patch_counts=counts)
                self.assertEqual((accepted, counts), (2, [2,2,1,1]))
                self.assertEqual([len(h) for h in seen], [2,1,1])
                clean, history, *_ = predict.call_args.args[-1]
                expected = torch.cat((z[:,20:28].reshape(4,2,4), z[:,36:40].reshape(2,2,4)))
                torch.testing.assert_close(clean, expected)
                for i, value in ((1,4.), (2,5.), (3,6.), (5,4.)):
                    self.assertTrue((history[i, -2:] == value).all())
                self.assertFalse(history.requires_grad)
                self.assertTrue(core.training)
                loss.backward()
                self.assertTrue(any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.any()
                                    for p in core.backbone_llm.parameters()))

            with torch.no_grad():
                core.stop_head.bias.fill_(12.)
            with patch('random.Random.randint', return_value=10), \
                    patch.object(recovery, 'sample_native_patches', side_effect=AssertionError('sampled after STOP')):
                counts = [0]*4
                recovery.recovery_loss(tts, [ex], [{'audio':src}], 3, config=recovery.FOUR_PATCH, patch_counts=counts)
                self.assertEqual(counts, [1,0,0,0])

    def test_probe_prefills_both_histories_and_scores_only_free_suffix(self):
        core = tiny_core().eval()
        with torch.no_grad():
            core.stop_head.weight.zero_()
            core.stop_head.bias.fill_(-12.)
        z = torch.randn(1, 16, 4)
        history = dict(reference=z[:, :4], speaker=torch.randn(1,6), prefixes={'clean':z, 'metal':z+1},
                       prefix_samples=15, left=1, rate=100)
        tts = SimpleNamespace(tts_core=core, redae=SimpleNamespace(downsample_rate=1,
                                         decode=lambda latents: (latents[:, :, 0], 100)))
        fed, embeds = [], []
        def flow(hist_latents, backbone_cond, **kwargs):
            fed.append(hist_latents.clone())
            return torch.randn(1,2,4)
        handle = core.patch_encoder.register_forward_pre_hook(lambda _, inputs: embeds.append(inputs[0].clone()))
        try:
            with patch.object(probe, 'MAX_PATCHES', 10), patch.object(core, '_flow_one_step', flow):
                a, reason = probe.continuation(tts, history, torch.tensor([[1,2]]), 'clean', 1234)
                b, _ = probe.continuation(tts, history, torch.tensor([[1,2]]), 'metal', 1234)
            self.assertEqual((len(a), reason), (15+4, 'max_patches'))
            np.testing.assert_allclose(a[15:], b[15:])  # same noise in the mock sampler
            torch.testing.assert_close(fed[0], z[:, -core.history_length:])
            torch.testing.assert_close(fed[2], (z+1)[:, -core.history_length:])
            torch.testing.assert_close(embeds[0], torch.cat([z[:,:4], z], 1))
            torch.testing.assert_close(embeds[3], torch.cat([z[:,:4], z+1], 1))
            with torch.no_grad():
                core.stop_head.bias.fill_(12.)
            early, reason = probe.continuation(tts, history, torch.tensor([[1,2]]), 'clean', 1234)
            self.assertEqual((len(early), reason), (15, 'stop'))
            self.assertEqual(probe.score_suffix(early[15:], 100, 'Тест.', None)['asr']['wer'], 1.)
        finally:
            handle.remove()

    def test_new_run_starts_at_base_and_only_trains_100_then_500(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/sweep.vox_recipe_levels.REPORT).write_text('{}')
            report = root/'pilot.json'
            report.write_text(json.dumps(dict(asr='asr', speaker_model='speaker')))
            stages, trainings, audio = [], [], []
            def train(_, run, mode, step, resume, *, fork_from, recovery, shared_features):
                self.assertEqual((mode, resume, fork_from, recovery),
                                 ('lora', step == 500, None, sweep.vox_recovery.FOUR_PATCH))
                if resume:
                    self.assertTrue((run/'lora'/'step-0100'/'checkpoint.json').is_file())
                folder = run/'lora'/f'step-{step:04d}'
                folder.mkdir(parents=True)
                recipe = dict(sweep.transfer_recipe('lora','data'), acoustic_recovery=recovery, recovery_start_step=0)
                (folder/'checkpoint.json').write_text(json.dumps(dict(step=step,recipe=recipe)))
                trainings.append(step)
            def compare(_, run, step, *args, reuse_base, **kwargs):
                self.assertTrue(reuse_base)
                audio.append(step)
                return Path(tempfile.mkdtemp(dir=run))/'report.json'
            def listen(root, label, adapter, voice, recipe):
                stages.append(label)
                if len(stages)==2:
                    raise RuntimeError('interrupted probe')
            with patch.object(sweep, 'paired_data', return_value=({}, [], {'selection_sha256':'data'})), \
                    patch.object(probe, 'select_voice', return_value={'prefix': 'held'}), \
                    patch.object(sweep, 'train', train), patch.object(sweep, 'compare', compare), \
                    patch.object(sweep, 'load_saved_base'), patch.object(probe, 'probe_stage', listen):
                with self.assertRaisesRegex(RuntimeError, 'interrupted'):
                    sweep.fresh_recovery(root, root, [report], four_patch=True)
                run = sweep.fresh_recovery(root, root, [report], four_patch=True)
                self.assertEqual(trainings, [100,500])
                self.assertEqual(audio, [100,500])
                self.assertEqual(stages, ['base','step-0100','base','step-0100','step-0500'])
                self.assertEqual(json.loads((run/'replay.json').read_text())['steps'], [100,500])

    def test_probe_writes_matched_wavs_scores_suffix_and_reuses_completed_pairs(self):
        rate = 16000
        ref_word = SimpleNamespace(word='Да', start=.01, end=.10)
        def recognize(audio, **settings):
            self.assertEqual(len(audio), 2400)
            return [SimpleNamespace(text='Да.', words=[ref_word])], None
        fake_asr = SimpleNamespace(transcribe=recognize)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root/'reference.wav'
            src = np.sin(np.arange(rate)*.04).astype('float32') * .2
            sf.write(path, src, rate, subtype='FLOAT')
            info = dict(audio=str(path), text='Согласна.', audio_sha256=probe.file_digest(path), speaker_id='d')
            other = root/'opening.wav'
            sf.write(other, src[50:], rate, subtype='FLOAT')
            voice = dict(reference=info, prefix=dict(info, audio=str(other), audio_sha256=probe.file_digest(other)))
            tts = SimpleNamespace(device='cpu', tts_core=SimpleNamespace(patch_size=2, eval=lambda: None),
                                  _apply_frontend=lambda **kw: (kw['text'], 'Russian', [kw['text']]),
                                  _tokenize_text=lambda text: torch.tensor([[1,2]]))
            tts.redae = SimpleNamespace(sample_rate=rate, downsample_rate=40, eval=lambda: None,
                    pad_to_multiple_of=lambda a,m: torch.nn.functional.pad(a, ((-a.shape[1])%m,0)),
                    encode=lambda a,s: a[:,::40,None], decode=lambda z: (z[:,:,0].repeat_interleave(40,1), rate))
            tts.spk_extractor = torch.nn.Identity()
            with patch.object(tts.spk_extractor, 'forward', lambda wav, sr: wav.mean().reshape(1,1)):
                def generate(prompt_latents, **kw):
                    tail = torch.linspace(-.4,.4,60).reshape(1,-1,1)
                    return torch.cat([prompt_latents, tail],1)
                tts.tts_core.generate = generate
                manager = SimpleNamespace(load_base=lambda **kw: tts, unload=lambda: None)
                recipe = dict(base_sha256='a',dataset_sha256='b')
                with patch.object(probe,'ModelManager', return_value=manager), \
                        patch('faster_whisper.WhisperModel', return_value=fake_asr) as load_asr, \
                        patch.object(probe,'TEXTS',{'short':'Да.'}), patch.object(probe,'SEEDS',(7,)):
                    report = probe.probe_stage(root/'test','base',None,voice,recipe)
                    probe.probe_stage(root/'test','base',None,voice,recipe)
                    self.assertEqual(load_asr.call_count, 1)
                    result = json.loads(report.read_text())['pairs'][0]
                    self.assertEqual(result['wer_delta'], 0.)
                    self.assertEqual(result['matched_words'][0]['spectral_shape_distance_db'], 0.)
                    self.assertEqual(result['prefix_seconds'], len(src[50:])/rate)
                    for variant in ('clean','metal'):
                        file = root/'test'/result['variants'][variant]['audio']
                        self.assertEqual(sf.info(file).frames, len(src[50:])+2400)

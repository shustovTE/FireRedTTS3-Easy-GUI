"""Check batched math against the previous scalar graph, not another batch call."""
from dataclasses import replace
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

from fireredtts3_local.training_batch import (Example, batch_flow_loss, sample_flow,
    set_training, batch_conditioning, patch_embeddings)
from fireredtts3_local.training_objective import adapter_targets, conditioning, flow_prediction
from fireredtts3_local.training_features import (BatchFeatures, BatchRequest, FeatureCache,
    load_feature_pipeline, precompute_features)
from fireredtts3_local.training_run import TrainingOptions, planned_batches, validate_model
from fireredtts3_local.training_data import file_digest, load_manifest, SpeakerPairs
from test_russian_training import tts, Manager
import test_russian_training as toy


def tiny_core(layers=2):
    from fireredtts3_local.model_manager import ModelManager
    ModelManager._ensure_upstream_namespace()
    import fireredtts3.llm.fireredtts3_base as upstream
    cfg = dict(upstream.Qwen3_1_7B_ConfigDict)
    cfg.update(hidden_size=32, intermediate_size=64, num_hidden_layers=layers,
               num_attention_heads=4, num_key_value_heads=2, head_dim=8,
               vocab_size=64, attn_implementation='sdpa')
    with patch.object(upstream, 'Qwen3_1_7B_ConfigDict', cfg):
        return upstream.FireRedTTS3BaseCore(upstream.FireRedTTS3BaseCoreConfig(
            redae_dim=4, patch_size=2, spk_in_dim=6, patch_encoder_hidden_size=16,
            patch_encoder_depth=1, patch_encoder_num_heads=2,
            dit_hidden_size=16, dit_depth=1, dit_num_heads=2))


def scalar_reference(core, example, sample, stop_weight=.05):
    """Original separate-example conditioning and original flow prediction."""
    current = example.target.reshape(-1, core.patch_size, example.target.shape[-1])[sample.indices]
    history = F.pad(torch.cat([example.prompt, example.target], 1), (0, 0, core.history_length, 0))
    offsets = example.prompt.shape[1] + sample.indices[:, None]*core.patch_size + torch.arange(core.history_length)[None]
    noisy = (1-sample.times[:, None, None])*sample.noise + sample.times[:, None, None]*current
    def forward():
        cond, spk, stop = conditioning(core, example.tokens[None], example.prompt, example.target, example.speaker)
        pred = flow_prediction(core, cond, spk, history[0, offsets], noisy, sample.times, sample.indices, sample.drop)
        return pred, stop
    teacher = None
    if example.preservation_weight:
        with torch.no_grad(), core.disable_adapter():
            teacher = forward()
    pred, stop = forward()
    targets = torch.zeros_like(stop); targets[-1] = 1
    loss = F.mse_loss(pred, current-sample.noise) + stop_weight*F.binary_cross_entropy_with_logits(stop, targets)
    if teacher:
        loss += example.preservation_weight*(F.mse_loss(pred, teacher[0]) + stop_weight*F.mse_loss(stop.sigmoid(), teacher[1].sigmoid()))
    return loss


class BatchMathTests(unittest.TestCase):
    def setUp(self):
        self.threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, self.threads)
        torch.manual_seed(2026)
        from peft import LoraConfig, get_peft_model
        core = tiny_core()
        self.core = get_peft_model(core, LoraConfig(r=2, lora_alpha=4, lora_dropout=0,
                                                  target_modules=adapter_targets(core, 'backbone_dit')))
        # Nonzero adapters exercise actual retention, not identically zero teacher loss.
        with torch.no_grad():
            for name, parameter in self.core.named_parameters():
                if 'lora_B' in name:
                    parameter.normal_(std=.03)
        self.core.eval()
        self.tts = SimpleNamespace(tts_core=self.core, device=torch.device('cpu'))
        self.examples = [Example(torch.arange(1, text+1), torch.randn(1, prompt*2, 4),
                                 torch.randn(1, target*2, 4), torch.randn(1, 6), keep)
                         for text, prompt, target, keep in [(2,1,1,.1), (7,3,4,0.), (1,2,2,.7)]]
        self.samples = [sample_flow(e, 2, 3, .35) for e in self.examples]

    def test_batch_matches_original_scalar_losses_and_gradients(self):
        expected = sum(scalar_reference(self.core, e, s) for e, s in zip(self.examples, self.samples))/3
        expected.backward()
        gradients = {n: p.grad.clone() for n,p in self.core.named_parameters() if p.grad is not None}
        self.core.zero_grad(set_to_none=True)
        shapes = []
        handle = self.core.backbone_llm.register_forward_pre_hook(lambda m, a, kw: shapes.append(tuple(kw['inputs_embeds'].shape)), with_kwargs=True)
        actual, metrics = batch_flow_loss(self.tts, self.examples, samples=self.samples)
        handle.remove()
        # One retained-subset teacher and ONE batched student backbone call.
        self.assertEqual([shape[0] for shape in shapes], [2, 3])
        self.assertGreater(float(metrics['preservation']), 0)
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
        actual.backward()
        for name, parameter in self.core.named_parameters():
            if name in gradients:
                torch.testing.assert_close(parameter.grad, gradients[name], atol=3e-6, rtol=3e-4, msg=name)
            else:
                self.assertIsNone(parameter.grad, name)
        for prefix in ('backbone_llm', 'dit.'):
            self.assertTrue(any(g.abs().sum() > 0 for n,g in gradients.items() if prefix in n))

    def test_masks_history_stop_and_no_cross_example_effect(self):
        embeds = patch_embeddings(self.core, self.examples)
        cond, _, stops, targets, stop_owners, patch_owners = batch_conditioning(self.core, self.examples, self.samples, embeds)
        for i, (e,s) in enumerate(zip(self.examples, self.samples)):
            scalar_cond, _, scalar_stop = conditioning(self.core, e.tokens[None], e.prompt, e.target, e.speaker)
            torch.testing.assert_close(cond[patch_owners==i], scalar_cond[s.indices], atol=2e-6, rtol=2e-5)
            torch.testing.assert_close(stops[stop_owners==i], scalar_stop, atol=2e-6, rtol=2e-5)
            self.assertEqual(targets[stop_owners==i].tolist(), [0.]*(e.target.shape[1]//2)+[1.])
        changed = list(self.examples)
        changed[1] = replace(changed[1], tokens=torch.arange(1,30), target=changed[1].target*100)
        cond2, _, stops2, _, owners2, patch2 = batch_conditioning(self.core, changed, self.samples, patch_embeddings(self.core, changed))
        torch.testing.assert_close(cond[patch_owners==0], cond2[patch2==0], atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(stops[stop_owners==0], stops2[owners2==0], atol=2e-6, rtol=2e-5)

    def test_precomputed_patches_match_online_with_checkpointing_and_dropout(self):
        self.core.backbone_llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        set_training(self.core)
        self.assertFalse(self.core.patch_encoder.training)
        with torch.no_grad():
            embeddings = patch_embeddings(self.core, self.examples)
        cached = [replace(e, **emb) for e,emb in zip(self.examples, embeddings)]
        state = torch.get_rng_state()
        online, _ = batch_flow_loss(self.tts, self.examples, samples=self.samples)
        online.backward()
        gradients = {n:p.grad.clone() for n,p in self.core.named_parameters() if p.grad is not None}
        self.core.zero_grad(set_to_none=True)
        torch.set_rng_state(state)
        with patch.object(self.core.patch_encoder, 'forward', side_effect=AssertionError('cache must bypass PatchEncoder')):
            actual, _ = batch_flow_loss(self.tts, cached, samples=self.samples)
            actual.backward()
        torch.testing.assert_close(actual, online, rtol=0, atol=0)
        for n,p in self.core.named_parameters():
            if n in gradients:
                torch.testing.assert_close(p.grad, gradients[n], rtol=0, atol=0)
        self.assertTrue(self.core.training)
        self.assertFalse(self.core.patch_encoder.training)

    def test_float32_cache_restores_bfloat16_encoder_output_dtype(self):
        self.core.to(torch.bfloat16)
        def amp(_):
            return torch.autocast('cpu', dtype=torch.bfloat16)
        with torch.no_grad(), amp(None):
            embeddings = patch_embeddings(self.core, self.examples)
        cached = [replace(e, **{key:value.float() for key,value in emb.items()})
                  for e,emb in zip(self.examples,embeddings)]
        with patch('fireredtts3_local.training_batch.autocast', side_effect=amp):
            online, _ = batch_flow_loss(self.tts,self.examples,samples=self.samples)
            actual, _ = batch_flow_loss(self.tts,cached,samples=self.samples)
        torch.testing.assert_close(actual,online,rtol=0,atol=0)

    def test_feature_only_loader_uses_actual_patch_weights_without_backbone(self):
        import fireredtts3.llm.fireredtts3_base as upstream
        import fireredtts3.redae.redae as redae_module
        import fireredtts3.campp.campp as campp_module
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            original = self.core.get_base_model()
            # Save a clean non-PEFT base for the feature-only loader.
            original = tiny_core().eval()
            original.save_pretrained(base/'fireredtts3_base')
            with (patch.object(upstream, 'FireRedTTS3BaseCore', side_effect=AssertionError('no backbone')),
                 patch.object(redae_module.RedAE, 'from_pretrained', return_value=torch.nn.Linear(1,1)),
                 patch.object(campp_module, 'CamppEmbedding', return_value=torch.nn.Linear(1,1))):
                pipeline = load_feature_pipeline(base, 'cpu')
            x = torch.randn(1,6,4)
            torch.testing.assert_close(pipeline.tts_core.patch_encoder(x), original.patch_encoder(x), rtol=0, atol=0)
            self.assertFalse(any(p.requires_grad for p in pipeline.tts_core.patch_encoder.parameters()))


class FeatureTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.tts = tts()
        self.tts.tts_core.patch_encoder.requires_grad_(False)
        self.rows = []
        for i in range(3):
            path = self.root/f'{i}.wav'
            sf.write(path, np.sin(np.arange(24000+i*4800)*(.1+i*.01))*.1, 24000)
            self.rows.append({'utt_id':str(i), 'audio':str(path), 'audio_sha256':file_digest(path),
                              'text':'Привет!', 'language':'Russian', 'speaker_id':'a', 'duration':1+i*.2})

    def test_precompute_reuse_repair_readonly_and_fingerprint(self):
        root = self.root/'cache'
        report = precompute_features(self.tts, self.rows+self.rows, root, 'base', log=lambda _:None)
        self.assertEqual(report['encoded'], 3)
        cache = FeatureCache(self.tts, root, 'base', read_only=True)
        cache.require_all(self.rows)
        with patch.object(self.tts.redae, 'encode', side_effect=AssertionError('must never encode')):
            self.assertEqual(precompute_features(self.tts, self.rows, root, 'base', log=lambda _:None)['reused'],3)
            cache.get_features(self.rows[0])
            with self.assertRaisesRegex(ValueError, 'missing'):
                FeatureCache(self.tts, root, 'different', read_only=True).require_all(self.rows)
        cache.path(self.rows[0]).write_bytes(b'broken')
        with self.assertRaisesRegex(ValueError, 'Invalid feature cache'):
            cache.read(self.rows[0])
        self.assertEqual(precompute_features(self.tts, self.rows, root, 'base', repair=True, log=lambda _:None)['encoded'],1)
        cache.path(self.rows[1]).unlink()
        with self.assertRaisesRegex(FileNotFoundError, 'precompute'):
            cache.get_features(self.rows[1])
        with self.assertRaisesRegex(RuntimeError, 'forbidden'):
            cache.encode(self.rows[2])

    def test_prefetch_order_and_validation_independent_of_batch_size(self):
        precompute_features(self.tts, self.rows, self.root/'cache', 'base', log=lambda _:None)
        cache = FeatureCache(self.tts, self.root/'cache', 'base', read_only=True)
        requests = [BatchRequest([(self.rows[i%3], self.rows[(i+1)%3],0.)], (i,)) for i in range(7)]
        loader = BatchFeatures(cache, workers=2, prefetch=2)
        try:
            result = list(loader.iterate(requests))
            self.assertEqual([r.sampler_state for r,_ in result], [(i,) for i in range(7)])
        finally:
            loader.close()
        # No GPU encoder invocation in either validation batch configuration.
        pairs = SpeakerPairs(self.rows)
        with patch.object(self.tts.redae, 'encode', side_effect=AssertionError('readonly')):
            a = validate_model(self.tts, cache, pairs, TrainingOptions(validation_pairs=7, validation_batch_size=1), 987)
            b = validate_model(self.tts, cache, pairs, TrainingOptions(validation_pairs=7, validation_batch_size=3), 987)
        for key in a:
            self.assertAlmostEqual(a[key], b[key], places=5)


class PlanningTests(unittest.TestCase):
    def test_bucketing_preserves_samples_and_resume_despite_prefetch(self):
        rows = [dict(utt_id=str(i), audio_sha256=str(i), speaker_id='a', language='Russian',
                     duration=1+i*.1, text='x'*i) for i in range(11)]
        train = SpeakerPairs(rows)
        replay = SpeakerPairs([dict(r, speaker_id='b', audio_sha256='b'+r['audio_sha256']) for r in rows])
        options = TrainingOptions(batch_size=3, replay_ratio=.4)
        state = random.Random(33).getstate()
        planned = list(planned_batches(train,replay,options,state,0,4,2,'Epochs'))
        unsorted = list(planned_batches(train,replay,replace(options,bucket_by_length=False),state,0,4,2,'Epochs'))
        def updates(batches):
            groups, pending = [], []
            for b in batches:
                pending.extend((p['audio_sha256'],t['audio_sha256'],keep) for p,t,keep in b.pairs)
                if b.sampler_state:
                    groups.append(sorted(pending)); pending=[]
            return groups
        self.assertEqual([len(g) for g in updates(planned)], [6,5,6,5])
        self.assertEqual(updates(planned),updates(unsorted))
        # The iterable may have prefetched later updates; resume from consumed state only.
        resume_state = planned[1].sampler_state
        resumed = list(planned_batches(train,replay,options,resume_state,1,4,2,'Epochs'))
        self.assertEqual(updates(resumed), updates(planned)[1:])

    def test_invalid_batch_options_fail_early(self):
        for options in ({'batch_size':0}, {'batch_size':True}, {'prefetch_batches':0}, {'loader_workers':-1},
                        {'require_precomputed_features':True,'cache_dir':''}, {'offload_frozen_encoders':True}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                TrainingOptions.parse(options)


class BatchedRunTests(unittest.TestCase):
    def test_unequal_microbatches_match_scalar_accumulation(self):
        fixture = toy.DataTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        from fireredtts3_local.training_run import train_lora
        from safetensors.torch import load_file
        options = TrainingOptions(batch_size=5, validation_pairs=2, validate_every=1,
                                  patience=0, gradient_checkpointing=False, bucket_by_length=False,
                                  cache_dir=str(fixture.root/'cache'), require_precomputed_features=True)
        torch.manual_seed(1234)
        pipeline = tts(); pipeline.tts_core.patch_encoder.requires_grad_(False)
        precompute_features(pipeline, load_manifest(fixture.manifest), options.cache_dir, 'toy', log=lambda _:None)
        kwargs = dict(rank=2, alpha=4, dropout=0., learning_rate=1e-3, length_mode='Epochs',
                      training_epochs=2, save_every=1, log=lambda _:None)
        with patch('fireredtts3_local.training_run.require_training_hardware'), patch('fireredtts3_local.training_run.model_identity',return_value='toy'):
            # Each epoch has 9 examples: batches 5+4 must have weights 5/9 and 4/9.
            train_lora(Manager(), fixture.manifest, fixture.root/'batch', gradient_accumulation=2, options=options, **kwargs)
            train_lora(Manager(), fixture.manifest, fixture.root/'scalar', gradient_accumulation=9, options=replace(options,batch_size=1), **kwargs)
        a = load_file(str(fixture.root/'batch'/'adapter_model.safetensors'))
        b = load_file(str(fixture.root/'scalar'/'adapter_model.safetensors'))
        for key in a:
            torch.testing.assert_close(a[key],b[key],rtol=1e-5,atol=1e-7)

    def test_cached_batched_resume_and_final_partial_update(self):
        # Reuse data creation, without inheriting and rerunning its test methods.
        fixture = toy.DataTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        from fireredtts3_local.training_run import train_lora
        from fireredtts3_local.training import STOP_EVENT
        from safetensors.torch import load_file
        # 9 train examples per epoch: effective 6 -> updates with 6 and 3 examples.
        opts = TrainingOptions(batch_size=3, validation_batch_size=2, validation_pairs=3,
                               validate_every=1, patience=0, gradient_checkpointing=False,
                               cache_dir=str(fixture.root/'cache'), require_precomputed_features=True,
                               offload_frozen_encoders=True)
        # Manager's toy base is initialized from seed 1234 in the actual runner.
        torch.manual_seed(1234)
        pipeline = tts(); pipeline.tts_core.patch_encoder.requires_grad_(False)
        precompute_features(pipeline, load_manifest(fixture.manifest), opts.cache_dir, 'toy', log=lambda _:None)
        kwargs = dict(rank=2, alpha=4, length_mode='Epochs', training_epochs=2,
                      gradient_accumulation=2, save_every=1, options=opts, log=lambda _:None)
        with patch('fireredtts3_local.training_run.require_training_hardware'), patch('fireredtts3_local.training_run.model_identity',return_value='toy'), patch('test_russian_training.Encoder.encode',side_effect=AssertionError('cache required')):
            train_lora(Manager(), fixture.manifest, fixture.root/'full', **kwargs)
            train_lora(Manager(), fixture.manifest, fixture.root/'resume', progress_cb=lambda *a:STOP_EVENT.set(), **kwargs)
            train_lora(Manager(), fixture.manifest, fixture.root/'resume', resume_checkpoint=fixture.root/'resume', **kwargs)
        a = load_file(str(fixture.root/'full'/'adapter_model.safetensors'))
        b = load_file(str(fixture.root/'resume'/'adapter_model.safetensors'))
        for key in a:
            torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)
        perf = [json.loads(line) for line in (fixture.root/'full'/'performance.jsonl').read_text().splitlines()]
        self.assertEqual([p['examples'] for p in perf],[6,3,6,3])

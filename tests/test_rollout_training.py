"""Causality, upstream ODE parity, real supervision and interrupted own-history training."""
import copy
from dataclasses import asdict, replace
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from peft import LoraConfig
from safetensors.torch import load_file
import soundfile as sf
import torch
import torch.nn.functional as F

from fireredtts3_local import training_rollout as rollout
from fireredtts3_local.hybrid import create_adapter, check_trainable
from fireredtts3_local.training_batch import Example, patch_embeddings, set_training
from fireredtts3_local.training_objective import adapter_targets
from fireredtts3_local.training_run import TrainingOptions, train_lora
from test_training_batch import tiny_core
import test_russian_training as toy


class RolloutMathTests(unittest.TestCase):
    def setUp(self):
        threads = torch.get_num_threads(); torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, threads)
        torch.manual_seed(222)
        core = tiny_core()
        self.core = create_adapter(core, LoraConfig(r=2, lora_alpha=4, lora_dropout=0,
            target_modules=adapter_targets(core, 'backbone')))
        with torch.no_grad():
            self.core.stop_head.weight.zero_(); self.core.stop_head.bias.fill_(-10)
        self.core.eval()
        self.tts = SimpleNamespace(tts_core=self.core)
        self.examples = [Example(torch.arange(1, 4+i), torch.randn(1, (i+1)*2, 4),
            torch.randn(1, 40+i*2, 4), torch.randn(1, 6), .1) for i in range(2)]
        self.options = rollout.RolloutOptions(warmup_steps=0, grow_every=1)

    def cached(self, examples):
        with torch.no_grad():
            return [replace(e, **emb) for e, emb in zip(examples, patch_embeddings(self.core, examples))]

    def test_prefix_conditioning_matches_native_prefill_for_unequal_lengths(self):
        prefixes = [replace(e, target=e.target[:, :n*2]) for e,n in zip(self.examples,(6,10))]
        with torch.no_grad():
            cond, spk, history, stop = rollout.prefix_conditioning(self.core, prefixes)
            for i, e in enumerate(prefixes):
                audio = torch.cat([e.prompt, e.target], 1)
                embedded = self.core.patch_encoder(audio)
                inputs = torch.cat([self.core.spk_proj_llm(e.speaker)[:, None],
                                    self.core.backbone_llm.embed_tokens(e.tokens[None]), embedded], 1)
                hidden, _ = self.core._backbone_one_step(inputs)
                native_condition = self.core.dit_head(hidden[:, -(self.core.history_patches+1):])
                torch.testing.assert_close(cond[i:i+1], native_condition, rtol=2e-5, atol=2e-6)
                torch.testing.assert_close(spk[i:i+1], self.core.spk_proj_dit(e.speaker), rtol=2e-6, atol=2e-8)
                torch.testing.assert_close(history[i:i+1], audio[:, -self.core.history_length:], rtol=0, atol=0)
                torch.testing.assert_close(stop[i], self.core.stop_head(hidden[:, -1]).float().flatten()[0])

    def test_batched_sampler_matches_native_euler_cfg_for_each_example(self):
        with torch.no_grad():
            cond, spk, history, _ = rollout.prefix_conditioning(self.core, self.examples)
            for cfg in (0., 2.):
                options = replace(self.options, inference_cfg=cfg)
                times = torch.linspace(0, 1, options.n_timesteps+1)
                times = 1-torch.cos(times*.5*torch.pi)
                noise = torch.randn(2, self.core.patch_size, 4)
                with patch('torch.randn', return_value=noise.clone()):
                    actual = rollout.sample_native_patches(self.core, cond, spk, history, options)
                for i in range(2):
                    with patch('torch.randn', return_value=noise[i:i+1].clone()):
                        expected = self.core._flow_one_step(history[i:i+1],cond[i:i+1],spk[i:i+1],times,cfg)
                    torch.testing.assert_close(actual[i:i+1], expected, rtol=2e-5, atol=2e-6)

    def test_future_audio_cannot_affect_rollout_and_own_cache_is_reencoded(self):
        examples = self.cached(self.examples)
        changed = [replace(e, target=torch.cat([e.target[:, :12], e.target[:, 12:]+100], 1),
                           target_embeds=None) for e in examples]
        changed = self.cached(changed)
        torch.manual_seed(8)
        first, attempted, rejected = rollout.collect_windows(self.core, examples, self.options, 4, anchors=[6,6])
        torch.manual_seed(8)
        other, _, _ = rollout.collect_windows(self.core, changed, self.options, 4, anchors=[6,6])
        self.assertEqual((len(first),attempted,rejected),(2,2,0))
        for i,(a,b) in enumerate(zip(first,other)):
            self.assertEqual(a.index,10)
            self.assertEqual(a.prefix.target.shape[1],20)
            self.assertFalse(a.prefix.target.requires_grad)
            self.assertFalse(a.prefix.target_embeds.requires_grad)
            torch.testing.assert_close(a.prefix.target,b.prefix.target,rtol=0,atol=0)
            torch.testing.assert_close(a.prefix.target_embeds,self.core.patch_encoder(a.prefix.target),rtol=2e-5,atol=2e-6)
            self.assertFalse(torch.equal(a.prefix.target[:,12:],examples[i].target[:,12:20]))
            # The cached source data was never edited in place.
            torch.testing.assert_close(examples[i].target,self.examples[i].target,rtol=0,atol=0)

    def test_real_targets_own_history_and_only_backbone_lora_receives_gradient(self):
        examples = self.cached(self.examples)
        windows, _, _ = rollout.collect_windows(self.core, examples, self.options, 2, anchors=[6,7])
        frozen = {n:p.clone() for n,p in self.core.named_parameters() if not p.requires_grad}
        observed = []
        native_predict = rollout.predict
        def observe(core,cond,spk,tensors,chunk_size):
            observed.append(tensors)
            return native_predict(core,cond,spk,tensors,chunk_size)
        set_training(self.core)
        with patch.object(rollout,'collect_windows',return_value=(windows,2,0)), \
             patch.object(rollout,'predict',side_effect=observe):
            loss, metrics = rollout.rollout_loss(self.tts, examples, self.options, 1, dit_chunk_size=1)
        loss.backward()
        self.assertGreater(float(loss.detach()),0.)
        self.assertEqual(metrics['rollout_accepted_fraction'],1.)
        current, history, _, _, drop = observed[0]
        for i,w in enumerate(windows):
            torch.testing.assert_close(current[i:i+1],examples[w.owner].target[:,w.index*2:(w.index+1)*2],rtol=0,atol=0)
            torch.testing.assert_close(history[i:i+1],w.prefix.target[:,-self.core.history_length:],rtol=0,atol=0)
        self.assertFalse(drop.any())
        named = check_trainable(self.core,'backbone')
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for _,p in named))
        self.assertTrue(all('backbone_llm.' in n for n,p in self.core.named_parameters() if p.grad is not None))
        torch.optim.AdamW([p for _,p in named],lr=.001).step()
        for n,p in self.core.named_parameters():
            if n in frozen:
                torch.testing.assert_close(p,frozen[n],rtol=0,atol=0)
                self.assertIsNone(p.grad)

    def test_terminal_state_is_checked_and_rejected_windows_are_not_targets(self):
        native = rollout.prefix_conditioning
        def stop_after_one(core, examples):
            cond,spk,hist,stop = native(core, examples)
            if examples[0].target.shape[1] >= 14:
                stop = torch.full_like(stop,10)
            return cond,spk,hist,stop
        set_training(self.core)
        with patch.object(rollout,'prefix_conditioning',side_effect=stop_after_one):
            windows,attempted,rejected = rollout.collect_windows(self.core,self.examples,self.options,1,anchors=[6,6])
        self.assertEqual((len(windows),attempted,rejected),(0,2,2))
        self.assertTrue(self.core.training)
        self.assertFalse(self.core.patch_encoder.training)
        with torch.no_grad():self.core.stop_head.bias.fill_(10)
        with patch.object(rollout,'predict',side_effect=AssertionError('no accepted targets')):
            loss,metrics=rollout.rollout_loss(self.tts,self.examples,self.options,2)
        self.assertFalse(loss.requires_grad)
        self.assertEqual(float(loss),0.)
        self.assertEqual(metrics['rollout_stop_rejected_fraction'],1.)

    def test_warmup_short_records_and_failure_restore_state(self):
        before = torch.get_rng_state()
        loss,metrics=rollout.rollout_loss(self.tts,self.examples,replace(self.options,warmup_steps=20),0)
        torch.testing.assert_close(torch.get_rng_state(),before,rtol=0,atol=0)
        self.assertEqual(metrics['rollout_horizon'],0.)
        self.assertEqual(float(loss),0.)
        short = [replace(e,target=e.target[:,:8],target_embeds=None) for e in self.examples]
        loss,metrics=rollout.rollout_loss(self.tts,short,self.options,1)
        self.assertEqual(metrics['rollout_attempted_fraction'],0.)
        set_training(self.core)
        with patch.object(rollout,'sample_native_patches',side_effect=FloatingPointError('test')):
            with self.assertRaises(FloatingPointError):
                rollout.collect_windows(self.core,self.examples,self.options,1)
        self.assertTrue(self.core.training)
        self.assertFalse(self.core.patch_encoder.training)


class RolloutRunTests(unittest.TestCase):
    def test_resume_cli_keeps_objective_and_refuses_previous_hybrid_stage(self):
        import tempfile
        from fireredtts3_local.rollout_training import resume_recipe
        repo=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); job=root/'job';job.mkdir();run=root/'run';run.mkdir()
            (run/'trainer_state.pt').touch()
            config=json.loads((repo/'recipes/russian_rollout_96gb.json').read_text())
            config['output_dir']=str(run)
            (job/'recipe.json').write_text(json.dumps(config))
            with patch('fireredtts3_local.training_job.read_status',return_value=dict(active=False,status='completed')):
                resume_recipe(job,root/'resume.json')
                resumed=json.loads((root/'resume.json').read_text())
                self.assertEqual(resumed,{**config,'resume_checkpoint':str(run)})
                config['options']['profile']='hybrid';(job/'recipe.json').write_text(json.dumps(config))
                with self.assertRaises(ValueError):resume_recipe(job,root/'bad.json')

    def test_resume_restores_rng_optimizer_curriculum_and_real_rollouts_exactly(self):
        fixture=toy.DataTests(); fixture.setUp(); self.addCleanup(fixture.doCleanups)
        root=fixture.root
        class Manager(toy.Manager):
            def load_base(self,**kwargs):
                result=super().load_base(**kwargs)
                with torch.no_grad():
                    result.tts_core.stop_head.weight.zero_();result.tts_core.stop_head.bias.fill_(-10)
                return result
        options=TrainingOptions(profile='backbone',precision_policy='native',gradient_checkpointing=False,
            rollout=asdict(rollout.RolloutOptions(warmup_steps=1,grow_every=1,n_timesteps=2)),
            batch_size=3,validation_pairs=3,validation_batch_size=2,validate_every=2,patience=0,
            patches_per_example=3,dit_chunk_size=2,cache_dir=str(root/'cache'))
        kwargs=dict(rank=2,alpha=4,dropout=0.,training_steps=4,gradient_accumulation=1,
                    save_every=2,options=options,log=lambda _:None)
        from fireredtts3_local.training import STOP_EVENT
        def stop_at_two(step,*_):
            if step==2:STOP_EVENT.set()
        with patch('fireredtts3_local.training_run.require_training_hardware'), \
             patch('fireredtts3_local.training_run.model_identity',return_value='toy'), \
             patch.object(toy.Encoder,'downsample_rate',240):
            train_lora(Manager(),fixture.manifest,root/'full',**kwargs)
            train_lora(Manager(),fixture.manifest,root/'paused',progress_cb=stop_at_two,**kwargs)
            train_lora(Manager(),fixture.manifest,root/'paused',resume_checkpoint=root/'paused',**kwargs)
        for name,value in load_file(root/'full/adapter_model.safetensors').items():
            torch.testing.assert_close(value,load_file(root/'paused/adapter_model.safetensors')[name],rtol=0,atol=0)
        a=[json.loads(s) for s in (root/'full/rollout_metrics.jsonl').read_text().splitlines()]
        b=[json.loads(s) for s in (root/'paused/rollout_metrics.jsonl').read_text().splitlines()]
        self.assertEqual(a,b)
        self.assertEqual([r['rollout_horizon'] for r in a],[0.,1.,2.,4.])
        self.assertTrue(all(r['rollout_contribution']>0 for r in a[1:]), a)
        # Retention is evaluated only by the unchanged clean-data branch.
        state=torch.load(root/'paused/trainer_state.pt',weights_only=True)
        self.assertEqual(state['recipe']['rollout_implementation']['schema'],rollout.SCHEMA)
        self.assertEqual(state['step'],4)

    def test_options_and_curriculum(self):
        options=rollout.RolloutOptions()
        self.assertEqual([options.horizon(n) for n in (0,19,20,119,120,219,220,999)],[0,0,1,1,2,2,4,4])
        for value in ({'max_history_patches':20},{'min_prefix_patches':0},{'warmup_steps':-1},
                      {'weight':float('nan')},{'stop_threshold':1.},{'grow_every':True}):
            with self.assertRaises(ValueError):rollout.RolloutOptions.parse(value)
        with self.assertRaisesRegex(ValueError,'native precision'):
            TrainingOptions(rollout={}).validate()
        with self.assertRaisesRegex(ValueError,'native precision'):
            TrainingOptions(profile='hybrid',precision_policy='native',rollout={}).validate()

    def test_preparation_keeps_original_audio_and_never_loads_old_adapter(self):
        from test_common_voice_spontaneous import PreparationTests, REPO
        from fireredtts3_local.common_voice_spontaneous import prepare as prepare_cv
        from fireredtts3_local.rollout_training import prepare,audit
        fixture=PreparationTests();fixture.setUp();self.addCleanup(fixture.doCleanups)
        prepare_cv(fixture.archive(),fixture.output,fixture.dialogs,fixture.recipe,log=lambda _:None)
        shutil.rmtree(fixture.output/'initial-adapter');shutil.rmtree(fixture.adapter)
        config=json.loads((REPO/'recipes/russian_rollout_96gb.json').read_text())
        reference=fixture.root/'voice.wav';sf.write(reference,np.sin(np.arange(24000)*.133)*.1,24000)
        benchmark=fixture.root/'benchmark.jsonl'
        benchmark.write_text(json.dumps(dict(id='one',reference_audio=str(reference),reference_text='Референс.',
                                            text='Новый вопрос?',seed=1234))+'\n')
        config['output_dir']=str(fixture.root/'rollout-run')
        config['options']['generation_benchmark']=str(benchmark)
        config['options']['cache_dir']=str(fixture.root/'cache')
        recipe=fixture.root/'recipe.json';recipe.write_text(json.dumps(config))
        output=fixture.root/'rollout-data'
        report=prepare(fixture.dialogs,fixture.output,output,recipe)
        self.assertEqual(report,audit(output))
        self.assertEqual(report['schema'],'russian-joint-rollout-v1')
        self.assertEqual([report['datasets'][name]['clips'] for name in
                         ('train','dialogs_validation','spontaneous_validation')],[51,9,6])
        self.assertEqual(report['maximum_steps'],35)
        self.assertEqual(report['audio'],'existing full recordings, no new cropping or normalization')
        prepared=json.loads((output/'recipe.json').read_text())
        self.assertIsNone(prepared['initial_adapter'])
        self.assertEqual(prepared['options']['profile'],'backbone')
        self.assertEqual(prepared['options']['rollout']['max_history_patches'],4)

    def test_short_whole_utterance_level_drift_cannot_hide_in_overlapping_windows(self):
        import tempfile
        from fireredtts3_local.training_monitor import load_benchmark,monitor
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);ref=root/'ref.wav';sf.write(ref,np.zeros(2400),24000)
            benchmark=root/'benchmark.jsonl'
            benchmark.write_text(json.dumps(dict(id='one',reference_audio=str(ref),reference_text='Речь.',text='Вопрос?'))+'\n')
            _,signature=load_benchmark(benchmark)
            tts=toy.tts();set_training(tts.tts_core)
            amplitude=torch.ones(24000*6)*.1
            def generate(**kwargs):
                self.assertFalse(kwargs['do_split'])
                for _ in range(7):tts.tts_core.stop_head(torch.zeros(1,8))
                return amplitude[None],24000
            tts.generate=generate
            with torch.no_grad():tts.tts_core.stop_head.bias.fill_(3.)
            first=monitor(tts,benchmark,root,0,signature,log=lambda _:None,nonoverlapping_windows=True)
            amplitude=torch.linspace(.1,.4,len(amplitude))
            second=monitor(tts,benchmark,root,1,signature,log=lambda _:None,nonoverlapping_windows=True)
            self.assertTrue(first['passed'])
            self.assertIn('level_drift_above_baseline_plus_3db',second['failures']['one'])
            report=json.loads(Path(second['report']).read_text())
            self.assertEqual(report['results'][0]['rms_window_seconds'],2.)
            with self.assertRaisesRegex(ValueError,'measurement changed'):
                monitor(tts,benchmark,root,2,signature,log=lambda _:None)


if __name__=='__main__':unittest.main()

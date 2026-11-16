"""Exercise dense acoustics, frozen teacher, exact resume, and actual PEFT export."""
import copy
from dataclasses import replace
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from peft import LoraConfig, get_peft_model_state_dict
from safetensors.torch import load_file
import soundfile as sf
import torch

from fireredtts3_local.hybrid import check_trainable, create_adapter, load_adapter
from fireredtts3_local.training_batch import (Example, batch_conditioning, batch_flow_loss,
    patch_embeddings, reduce_stop_loss, sample_flow, set_training)
from fireredtts3_local.training_features import FeatureCache
from fireredtts3_local.training_objective import adapter_targets, dense_targets
from fireredtts3_local.training_run import TrainingOptions, initial_adapter_info, train_lora
from test_training_batch import tiny_core
import test_russian_training as toy


def hybrid(core):
    return create_adapter(core, LoraConfig(r=2, lora_alpha=4, lora_dropout=0.,
        target_modules=adapter_targets(core, 'hybrid'), modules_to_save=list(dense_targets('hybrid'))))


class HybridMathTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(71)
        self.original = tiny_core()
        self.core = hybrid(copy.deepcopy(self.original))
        self.tts = SimpleNamespace(tts_core=self.core, device='cpu')
        self.examples = [Example(torch.arange(1, 4), torch.randn(1, 4, 4),
            torch.randn(1, 8, 4), torch.randn(1, 6), .1)]
        self.samples = [sample_flow(e, 2, 'all', 0) for e in self.examples]

    def test_dense_updates_teacher_unchanged_checkpointed_backward_and_reload(self):
        named = check_trainable(self.core, 'hybrid')
        frozen = {n: p.detach().clone() for n,p in self.core.named_parameters() if not p.requires_grad}
        originals = {name: copy.deepcopy(getattr(self.core, name).original_module.state_dict())
                     for name in dense_targets('hybrid')}
        optimizer = torch.optim.AdamW([p for _,p in named], lr=.001)
        set_training(self.core)
        for _ in range(2):
            optimizer.zero_grad()
            loss, metrics = batch_flow_loss(self.tts, self.examples, samples=self.samples, dit_chunk_size=2,
                precision_policy='native', stop_loss_mode='balanced', stop_gradient_to_backbone=False)
            loss.backward()
            for module in ('backbone_llm', 'dit', 'dit_head', 'stop_head'):
                self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for n,p in named
                                    if f'.{module}.' in n), module)
            optimizer.step()
        self.assertGreater(float(metrics['preservation']), 0.)
        for n,p in self.core.named_parameters():
            if n in frozen:
                torch.testing.assert_close(p, frozen[n], rtol=0, atol=0, msg=n)
                self.assertIsNone(p.grad)
        self.core.eval()
        for name in dense_targets('hybrid'):
            module = getattr(self.core, name)
            self.assertTrue(any(not torch.equal(value, originals[name][key])
                for key,value in module.modules_to_save['default'].state_dict().items()), name)
        with torch.no_grad(), self.core.disable_adapter():
            restored, _ = batch_flow_loss(self.tts, [replace(self.examples[0], preservation_weight=0)], samples=self.samples)
        clean, _ = batch_flow_loss(SimpleNamespace(tts_core=self.original.eval()),
            [replace(self.examples[0], preservation_weight=0)], samples=self.samples)
        torch.testing.assert_close(restored, clean, rtol=0, atol=0)
        with tempfile.TemporaryDirectory() as folder:
            self.core.save_pretrained(folder)
            Path(folder, 'firered_lora_metadata.json').write_text(json.dumps(
                {'base_fingerprint':'test', 'recipe':{'options':{'profile':'hybrid'}}}))
            initial_adapter_info(folder, 2, 4, 0., 'hybrid')
            loaded = load_adapter(copy.deepcopy(self.original), folder, is_trainable=True)
            check_trainable(loaded, 'hybrid')
            before, after = get_peft_model_state_dict(self.core), get_peft_model_state_dict(loaded)
            self.assertEqual(before.keys(), after.keys())
            for name in before:
                torch.testing.assert_close(before[name], after[name], rtol=0, atol=0)
            with loaded.disable_adapter():
                for name in dense_targets('hybrid'):
                    for key,value in getattr(loaded,name).original_module.state_dict().items():
                        torch.testing.assert_close(value, originals[name][key], rtol=0, atol=0)
            configuration = Path(folder, 'adapter_config.json')
            config = json.loads(configuration.read_text());config['modules_to_save'].remove('dit')
            configuration.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, 'dense modules'):
                initial_adapter_info(folder, 2, 4, 0., 'hybrid')

    def test_stop_loss_updates_head_without_backbone_gradient(self):
        set_training(self.core)
        embeddings = patch_embeddings(self.core, self.examples)
        _,_,logits,targets,owners,_ = batch_conditioning(self.core,self.examples,self.samples,embeddings,
                                                       'native',False)
        reduce_stop_loss(logits,targets,owners,1,'balanced').mean().backward()
        self.assertTrue(self.core.stop_head.modules_to_save['default'].weight.grad.abs().sum() > 0)
        self.assertTrue(all(p.grad is None for p in self.core.backbone_llm.parameters()))

    def test_terminal_gradient_no_longer_diluted_by_audio_length(self):
        gradients = {}
        for mode in ('mean','balanced'):
            gradients[mode] = []
            for count in (4,100):
                logits = torch.zeros(count+1, requires_grad=True)
                targets = torch.zeros(count+1);targets[-1] = 1
                reduce_stop_loss(logits, targets, torch.zeros(count+1,dtype=torch.long),1,mode).sum().backward()
                gradients[mode].append(float(logits.grad[-1]))
        self.assertEqual(gradients['balanced'],[-.25,-.25])
        self.assertGreater(abs(gradients['mean'][0]),abs(gradients['mean'][1])*10)

    def test_native_precision_boundary_and_cache_isolation(self):
        seen = {}
        hooks = [getattr(self.core,name).register_forward_hook(
            lambda _m,_a,value,n=name: seen.setdefault(n,[]).append(value.dtype))
            for name in ('patch_encoder','dit_head','dit','stop_head')]
        # CPU BF16 emulates the native backbone-only autocast boundary.
        try:
            with patch('fireredtts3_local.training_batch.autocast',
                       side_effect=lambda _:torch.autocast('cpu',dtype=torch.bfloat16)):
                loss,_ = batch_flow_loss(self.tts,self.examples,samples=self.samples,precision_policy='native')
                loss.backward()
        finally:
            for hook in hooks:hook.remove()
        for name,dtypes in seen.items():self.assertEqual(set(dtypes),{torch.float32},name)
        with tempfile.TemporaryDirectory() as folder:
            old = FeatureCache(self.tts,folder,'base')
            new = FeatureCache(self.tts,folder,'base',precision_policy='native')
            self.assertNotEqual(old.root,new.root)
            self.assertEqual(old.spec['version'],'redae-campp-patches-v2')
            self.assertEqual(new.spec['patch_precision'],'float32')
        self.core.patch_encoder.requires_grad_(True)
        with self.assertRaisesRegex(RuntimeError,'Trainable parameters'):
            check_trainable(self.core,'hybrid')


class HybridResumeTests(unittest.TestCase):
    def test_dense_optimizer_and_teacher_resume_exactly(self):
        fixture=toy.DataTests();fixture.setUp();self.addCleanup(fixture.doCleanups)
        root=fixture.root
        options=TrainingOptions(profile='hybrid',precision_policy='native',stop_loss_mode='balanced',
            stop_gradient_to_backbone=False,stop_lr_scale=10,keep_checkpoint_every=1,
            batch_size=3,validation_pairs=3,validation_batch_size=2,validate_every=2,patience=0,
            gradient_checkpointing=False,patches_per_example='all',dit_chunk_size=2)
        kwargs=dict(rank=2,alpha=4,dropout=0.,training_steps=4,gradient_accumulation=1,
                    save_every=2,options=options,log=lambda _:None)
        from fireredtts3_local.training import STOP_EVENT
        def stop_at_one(step,*_):
            if step == 1:STOP_EVENT.set()
        with patch('fireredtts3_local.training_run.require_training_hardware'), \
             patch('fireredtts3_local.training_run.model_identity',return_value='toy'):
            train_lora(toy.Manager(),fixture.manifest,root/'full',**kwargs)
            train_lora(toy.Manager(),fixture.manifest,root/'paused',progress_cb=stop_at_one,**kwargs)
            train_lora(toy.Manager(),fixture.manifest,root/'paused',resume_checkpoint=root/'paused',**kwargs)
        for name,value in load_file(root/'full/adapter_model.safetensors').items():
            torch.testing.assert_close(value,load_file(root/'paused/adapter_model.safetensors')[name],rtol=0,atol=0)
        state=torch.load(root/'paused/trainer_state.pt',weights_only=False)
        self.assertEqual(len(state['optimizer']['param_groups']),3)
        self.assertEqual(state['step'],4)
        self.assertFalse((root/'paused/adapter-step-000001/trainer_state.pt').exists())
        self.assertEqual(json.loads((root/'paused/adapter-step-000001/firered_lora_metadata.json').read_text())['step'],1)

    def test_better_loss_cannot_replace_best_when_generation_regresses(self):
        fixture=toy.DataTests();fixture.setUp();self.addCleanup(fixture.doCleanups)
        root=fixture.root;out=root/'guarded'
        benchmark=root/'benchmark.jsonl'
        benchmark.write_text(json.dumps(dict(id='one',reference_audio=fixture.rows[0]['audio'],
                                            reference_text='Референс.',text='Другой вопрос?'))+'\n')
        options=TrainingOptions(profile='hybrid',generation_benchmark=str(benchmark),keep_checkpoint_every=1,
            validate_every=2,validation_pairs=3,patience=0,gradient_checkpointing=False)
        def check(_tts,_benchmark,_output,step,*_):
            return dict(step=step,passed=step==0)
        diagnostics={key:0. for key in ('stop_terminal_recall_05','stop_early_any_05',
                                       'stop_terminal_recall_03','stop_early_any_03')}
        with patch('fireredtts3_local.training_run.require_training_hardware'), \
             patch('fireredtts3_local.training_run.model_identity',return_value='toy'), \
             patch('fireredtts3_local.training_monitor.monitor',side_effect=check), \
             patch('fireredtts3_local.training_run.validate_model',side_effect=[
                 dict(loss=2.,flow=1.,stop=1.,**diagnostics),dict(loss=1.,flow=.5,stop=.5,**diagnostics)]):
            train_lora(toy.Manager(),fixture.manifest,out,rank=2,alpha=4,training_steps=2,
                       gradient_accumulation=1,save_every=2,options=options,log=lambda _:None)
        best=json.loads((out/'best-validation/firered_lora_metadata.json').read_text())
        self.assertEqual(best['step'],0)
        first=json.loads((out/'adapter-step-000001/firered_lora_metadata.json').read_text())
        self.assertFalse(first['generation_checked_at_this_step'])


class HybridExportTests(unittest.TestCase):
    def test_real_export_preserves_dense_biases_convolutions_and_frozen_codec(self):
        from test_release import ReleaseTests
        from fireredtts3_local.export import export_merged
        fixture=ReleaseTests();fixture.setUp();self.addCleanup(fixture.doCleanups)
        from fireredtts3.llm.fireredtts3_base import FireRedTTS3BaseCore
        core=FireRedTTS3BaseCore.from_pretrained(fixture.base/'fireredtts3_base')
        model=hybrid(core)
        with torch.no_grad():
            for _,p in check_trainable(model,'hybrid'):p.add_(.001)
        expected={f'{name}.{key}':value.clone() for name in dense_targets('hybrid')
                  for key,value in getattr(model,name).modules_to_save['default'].state_dict().items()}
        model.save_pretrained(fixture.adapter)
        metadata=fixture.adapter/'firered_lora_metadata.json'
        meta=json.loads(metadata.read_text());meta['recipe']['options']['profile']='hybrid'
        metadata.write_text(json.dumps(meta))
        output=fixture.root/'hybrid-export'
        export_merged(fixture.base,fixture.adapter,fixture.template/'config.json',output,
                      tokenizer_dir=fixture.template,log=lambda _:None)
        actual=load_file(output/'model.safetensors')
        for key,value in expected.items():torch.testing.assert_close(actual[key],value,rtol=0,atol=0,msg=key)
        for key,value in fixture.original.items():
            if not key.startswith(('backbone_llm.','dit.','dit_head.','stop_head.')):
                torch.testing.assert_close(actual[key],value,rtol=0,atol=0,msg=key)
        report=json.loads((output/'export_provenance.json').read_text())
        self.assertEqual(report['dense_tensors_verified'],len(expected))
        fixture.build('hybrid-release')
        released=load_file(fixture.root/'hybrid-release/model.safetensors')
        for key,value in expected.items():torch.testing.assert_close(released[key],value,rtol=0,atol=0,msg=key)


class JointPreparationTests(unittest.TestCase):
    def test_union_preserves_selections_without_reading_previous_adapter(self):
        from test_common_voice_spontaneous import PreparationTests, REPO
        from fireredtts3_local.common_voice_spontaneous import prepare as prepare_cv
        from fireredtts3_local.joint_training import prepare,audit
        fixture=PreparationTests();fixture.setUp();self.addCleanup(fixture.doCleanups)
        prepare_cv(fixture.archive(),fixture.output,fixture.dialogs,fixture.recipe,log=lambda _:None)
        shutil.rmtree(fixture.output/'initial-adapter');shutil.rmtree(fixture.adapter)
        config=json.loads((REPO/'recipes/russian_joint_96gb.json').read_text())
        reference=fixture.root/'voice.wav'
        sf.write(reference,np.sin(np.arange(24000)*.133)*.1,24000)
        benchmark=fixture.root/'benchmark.jsonl'
        benchmark.write_text(json.dumps(dict(id='one',reference_audio=str(reference),reference_text='Референс.',
                                           text='Новый вопрос?',seed=1234))+'\n')
        config['output_dir']=str(fixture.root/'joint-run')
        config['options']['generation_benchmark']=str(benchmark)
        config['options']['cache_dir']=str(fixture.root/'joint-cache')
        recipe=fixture.root/'joint.json';recipe.write_text(json.dumps(config))
        output=fixture.root/'joint'
        report=prepare(fixture.dialogs,fixture.output,output,recipe)
        self.assertEqual(report,audit(output))
        self.assertEqual([report['datasets'][name]['clips'] for name in
                         ('train','dialogs_validation','spontaneous_validation')],[51,9,6])
        self.assertEqual(report['maximum_steps'],35)
        prepared=json.loads((output/'recipe.json').read_text())
        self.assertIsNone(prepared['initial_adapter'])
        self.assertEqual(prepared['options']['replay_ratio'],0)
        self.assertEqual(prepared['options']['profile'],'hybrid')
        sf.write(reference,np.zeros(24000),24000)
        with self.assertRaisesRegex(ValueError,'benchmark/reference changed'):audit(output)


class GenerationMonitorTests(unittest.TestCase):
    def test_limit_clipping_and_relative_drift_are_detected(self):
        from fireredtts3_local.training_monitor import measure,failures
        base=measure(np.ones(24000)*.1,24000,[.01]*6+[.9])
        self.assertEqual(base['finish_reason'],'stop');self.assertEqual(failures(base),[])
        limit=measure(np.ones(24000)*.1,24000,[.01]*400)
        self.assertIn('generation_limit',failures(limit,base))
        clipped=measure(np.ones(24000)*1.1,24000,[.01]*6+[.9])
        self.assertIn('raw_clipping_above_0.1_percent',failures(clipped,base))
        drift=measure(np.r_[np.ones(240000)*.1,np.ones(240000)*.2],24000,[.01]*6+[.9])
        self.assertIn('level_drift_above_baseline_plus_3db',failures(drift,base))

    def test_monitor_restores_training_mode_rng_and_writes_unclipped_float(self):
        from fireredtts3_local.training_monitor import load_benchmark,monitor
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);ref=root/'ref.wav';sf.write(ref,np.zeros(2400),24000)
            benchmark=root/'benchmark.jsonl'
            benchmark.write_text(json.dumps(dict(id='one',reference_audio=str(ref),reference_text='Речь.',text='Вопрос?'))+'\n')
            _,signature=load_benchmark(benchmark)
            tts=toy.tts();set_training(tts.tts_core)
            def generate(**kwargs):
                for _ in range(7):tts.tts_core.stop_head(torch.zeros(1,8))
                tts.tts_core.patch_encoder(torch.zeros(1,2,2))
                torch.rand(10)
                return torch.ones(1,2400)*1.2,24000
            tts.generate=generate
            with torch.no_grad():tts.tts_core.stop_head.bias.fill_(3.)
            state=torch.get_rng_state()
            summary=monitor(tts,benchmark,root,0,signature,True,log=lambda _:None)
            self.assertTrue(tts.tts_core.training)
            torch.testing.assert_close(state,torch.get_rng_state(),rtol=0,atol=0)
            self.assertFalse(summary['passed'])
            report=json.loads(Path(summary['report']).read_text());result=report['results'][0]
            raw,_=sf.read(Path(summary['report']).parent/result['float_audio'])
            listening,_=sf.read(Path(summary['report']).parent/result['audio'])
            self.assertGreater(raw.max(),1.);self.assertLessEqual(listening.max(),.981)
            baseline=(root/'generation_baseline.json').read_bytes()
            monitor(tts,benchmark,root,1,signature,log=lambda _:None)
            self.assertEqual(baseline,(root/'generation_baseline.json').read_bytes())
            self.assertFalse(tts.tts_core.stop_head._forward_hooks)


if __name__=='__main__':unittest.main()

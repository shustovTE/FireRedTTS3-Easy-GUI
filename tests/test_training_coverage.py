"""Coverage, all-patch objectives, and a new stage initialized from an adapter."""
from dataclasses import replace
import copy
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import load_file

from fireredtts3_local.training_batch import Example, sample_flow
from fireredtts3_local.training_data import SpeakerPairs
from fireredtts3_local.training_objective import adapter_targets, validate_adapter_targets
from fireredtts3_local.training_run import (TrainingOptions, initial_adapter_info,
    planned_batches, train_lora)
import test_russian_training as toy


class AdapterTargetTests(unittest.TestCase):
    def test_real_peft_compressed_targets_save_and_reload_28_layers(self):
        from test_training_batch import tiny_core
        from peft import LoraConfig, PeftModel, get_peft_model, get_peft_model_state_dict
        from peft.tuners.lora.layer import LoraLayer
        for profile in ('backbone', 'backbone_dit'):
            with self.subTest(profile=profile), tempfile.TemporaryDirectory() as directory:
                core = tiny_core(layers=28)
                clean = copy.deepcopy(core)
                targets = adapter_targets(core, profile)
                model = get_peft_model(core, LoraConfig(r=2,lora_alpha=4,lora_dropout=.05,target_modules=targets))
                with torch.no_grad():
                    for name, parameter in model.named_parameters():
                        if 'lora_B' in name:
                            parameter.normal_(std=.01)
                model.save_pretrained(directory)
                root = Path(directory)
                stored = json.loads((root/'adapter_config.json').read_text())['target_modules']
                # Exercise PEFT's real optimization threshold, without mocking it.
                self.assertLess(len(stored),len(targets))
                self.assertNotEqual(set(stored),set(targets))
                (root/'firered_lora_metadata.json').write_text(json.dumps({
                    'base_fingerprint':'toy', 'recipe':{'options':{'profile':profile}}}))
                initial = initial_adapter_info(root,2,4,.05,profile)
                self.assertEqual(validate_adapter_targets(clean,initial['targets'],targets),set(targets))
                loaded = PeftModel.from_pretrained(clean,directory,is_trainable=True)
                actual = {name for name, layer in loaded.get_base_model().named_modules() if isinstance(layer,LoraLayer)}
                self.assertEqual(actual,set(targets))
                before, after = get_peft_model_state_dict(model), get_peft_model_state_dict(loaded)
                self.assertEqual(before.keys(),after.keys())
                for name in before:
                    torch.testing.assert_close(before[name],after[name],rtol=0,atol=0)
                config = json.loads((root/'adapter_config.json').read_text())
                config['layers_to_transform'] = 0
                (root/'adapter_config.json').write_text(json.dumps(config))
                with self.assertRaisesRegex(ValueError,'Unsupported initial_adapter'):
                    initial_adapter_info(root,2,4,.05,profile)

    def test_target_resolution_rejects_extra_missing_and_unused_targets(self):
        core = toy.Core()
        targets = adapter_targets(core,'backbone')
        self.assertEqual(validate_adapter_targets(core,targets,targets),set(targets))
        self.assertEqual(validate_adapter_targets(core,['proj'],targets),set(targets))
        for stored, reason in [(['proj','stop_head'],'extra'), (['proj','linear'],'extra'),
                               (['proj','backbone_llm'],'extra'), (['stop_head'],'missing'),
                               (['proj','not_a_module'],'unmatched')]:
            with self.subTest(stored=stored), self.assertRaisesRegex(ValueError,reason):
                validate_adapter_targets(core,stored,targets)
        for invalid in ('all-linear',[],['']):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                validate_adapter_targets(core,invalid,targets)


class CoverageTests(unittest.TestCase):
    def setUp(self):
        rows = []
        for speaker, count in [('a', 2), ('b', 5), ('c', 10)]:
            for index in range(count):
                key = f'{speaker}-{index}'
                rows.append(dict(utt_id=key, audio_sha256=key, speaker_id=speaker,
                                 language='Russian', duration=1+index*.1, text='text '*index,
                                 source_id=speaker, start=index*3., end=index*3.+1))
        self.train = SpeakerPairs(rows)
        self.options = TrainingOptions(sampling_strategy='coverage', batch_size=3)

    def test_each_target_once_per_epoch_with_distinct_references(self):
        planned = list(planned_batches(self.train, None, self.options, random.Random(9).getstate(),
                                       0, 9, 2, 'Epochs', seed=91))
        epochs, targets = [], []
        step = 0
        for batch in planned:
            for prompt, target, keep in batch.pairs:
                self.assertEqual(prompt['speaker_id'], target['speaker_id'])
                self.assertTrue(self.train.distinct(prompt, target))
                self.assertEqual(keep, self.options.preservation_weight)
                targets.append(target['utt_id'])
            if batch.sampler_state:
                step += 1
                if step % 3 == 0:
                    self.assertCountEqual(targets, [r['utt_id'] for r in self.train.rows])
                    self.assertEqual(len(set(targets)), 17)
                    epochs.append(targets); targets = []
        self.assertEqual(len(epochs), 3)
        self.assertNotEqual(epochs[0], epochs[1])
        self.assertFalse(targets)
        self.assertEqual([len(b.pairs) for b in planned[:6]], [3,3,3,3,3,2])

    def test_resume_mid_epoch_and_at_boundary_reproduces_prefetched_pairs(self):
        full = list(planned_batches(self.train, None, self.options, random.Random(9).getstate(),
                                    0, 6, 2, 'Epochs', seed=91))
        for consumed_updates, batches in [(1,2), (2,4), (3,6), (4,8)]:
            resumed = list(planned_batches(self.train, None, self.options,
                full[batches-1].sampler_state, consumed_updates, 6, 2, 'Epochs', seed=91))
            self.assertEqual(resumed, full[batches:])

    def test_all_patches_includes_every_position_even_beyond_64(self):
        for count in (1, 4, 65, 125):
            example = Example(torch.tensor([1]), torch.randn(1,4,2),
                              torch.randn(1,count*2,2), torch.randn(1,3))
            sample = sample_flow(example, 2, 'all', .1)
            self.assertEqual(sample.indices.sort().values.tolist(), list(range(count)))
            self.assertEqual(sample.noise.shape, (count,2,2))

    def test_coverage_and_patch_options_reject_ambiguous_inputs(self):
        for data in ({'sampling_strategy':'unknown'}, {'sampling_strategy':'coverage','replay_manifest':'replay.jsonl'},
                     {'patches_per_example':0}, {'patches_per_example':True}, {'patches_per_example':1.5},
                     {'validation_patches_per_example':False}, {'validation_patches_per_example':'typo'},
                     {'validate_every':'typo'}, {'validate_every':True}):
            with self.subTest(data=data), self.assertRaises(ValueError):
                TrainingOptions.parse(data)
        with self.assertRaisesRegex(ValueError, 'Epochs'):
            list(planned_batches(self.train,None,self.options,random.Random(9).getstate(),0,1,1,'Steps'))
        with self.assertRaisesRegex(ValueError, 'no replay'):
            list(planned_batches(self.train,self.train,self.options,random.Random(9).getstate(),0,1,1,'Epochs'))
        self.assertEqual(TrainingOptions.parse({'patches_per_example':'all','validation_patches_per_example':8}).patches_per_example, 'all')


class WarmStartTests(unittest.TestCase):
    def setUp(self):
        self.fixture = toy.DataTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        self.source = self.root/'source'
        self.opts = TrainingOptions(batch_size=3, validation_pairs=3, validation_batch_size=2,
                                   validate_every=1, patience=0, gradient_checkpointing=False,
                                   patches_per_example=3, cache_dir=str(self.root/'cache'))
        for name, kwargs in [('require_training_hardware', {}), ('model_identity', {'return_value':'toy'})]:
            context = patch('fireredtts3_local.training_run.'+name, **kwargs)
            context.start(); self.addCleanup(context.stop)
        train_lora(toy.Manager(),self.fixture.manifest,self.source,rank=2,alpha=4,
                   training_steps=2,gradient_accumulation=1,save_every=1,options=self.opts,log=lambda _:None)

    def test_initial_weights_validation_and_resumed_coverage_match(self):
        from fireredtts3_local.training import STOP_EVENT
        options = replace(self.opts, sampling_strategy='coverage', patches_per_example='all', batch_size=4,
                          validation_patches_per_example=3, validate_every='epoch',
                          require_precomputed_features=True, offload_frozen_encoders=True)
        # Phase one may not have sampled every training file. Precompute all here.
        from fireredtts3_local.training_data import load_manifest
        from fireredtts3_local.training_features import precompute_features
        torch.manual_seed(1234)
        pipeline = toy.tts(); pipeline.tts_core.patch_encoder.requires_grad_(False)
        precompute_features(pipeline,load_manifest(self.fixture.manifest),options.cache_dir,'toy',log=lambda _:None)
        kwargs = dict(rank=2,alpha=4,length_mode='Epochs',training_epochs=2,
                      gradient_accumulation=1,save_every=1,learning_rate=5e-6,
                      options=options,initial_adapter=self.source,log=lambda _:None)
        def stop_at_two(step, *_):
            if step == 2:
                STOP_EVENT.set()
        train_lora(toy.Manager(),self.fixture.manifest,self.root/'paused',progress_cb=stop_at_two,**kwargs)
        initial = load_file(str(self.root/'paused'/'best-validation'/'adapter_model.safetensors'))
        source = load_file(str(self.source/'adapter_model.safetensors'))
        for key in source:
            torch.testing.assert_close(initial[key],source[key],rtol=0,atol=0)
        source_validation = [json.loads(line) for line in (self.source/'validation_metrics.jsonl').read_text().splitlines()][-1]
        initial_validation = json.loads((self.root/'paused'/'validation_metrics.jsonl').read_text().splitlines()[0])
        self.assertEqual(initial_validation['step'],0)
        for key in ('loss','flow','stop'):
            self.assertEqual(initial_validation[key],source_validation[key])
        state = torch.load(self.root/'paused'/'trainer_state.pt',weights_only=True)
        self.assertEqual(state['step'],2)
        self.assertEqual(state['scheduler']['last_epoch'],2)  # new stage, not source step + 2
        self.assertIn('initial_adapter_fingerprint',state['recipe'])
        with patch.object(toy.Encoder,'encode',side_effect=AssertionError('cache must be reused')):
            train_lora(toy.Manager(),self.fixture.manifest,self.root/'paused',resume_checkpoint=self.root/'paused',**kwargs)
            train_lora(toy.Manager(),self.fixture.manifest,self.root/'full',**kwargs)
        a = load_file(str(self.root/'full'/'adapter_model.safetensors'))
        b = load_file(str(self.root/'paused'/'adapter_model.safetensors'))
        for key in a:
            torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)
        reports = [json.loads(line) for line in (self.root/'paused'/'coverage.jsonl').read_text().splitlines()]
        self.assertEqual([(r['epoch'],r['targets']) for r in reports],[(1,9),(2,9)])
        validation_steps = [json.loads(line)['step'] for line in (self.root/'paused'/'validation_metrics.jsonl').read_text().splitlines()]
        self.assertEqual(validation_steps,[0,3,6])
        origin_metadata = self.source/'firered_lora_metadata.json'
        changed = json.loads(origin_metadata.read_text())
        changed['note'] = 'changed origin'
        origin_metadata.write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError,'Resume recipe/data differ'):
            train_lora(toy.Manager(),self.fixture.manifest,self.root/'full',resume_checkpoint=self.root/'full',**kwargs)

    def test_wrong_rank_profile_and_base_rejected(self):
        with self.assertRaisesRegex(ValueError,'rank/alpha/dropout'):
            initial_adapter_info(self.source,4,4,.05,'backbone')
        with self.assertRaisesRegex(ValueError,'profile/base'):
            initial_adapter_info(self.source,2,4,.05,'backbone_dit')
        metadata_path = self.source/'firered_lora_metadata.json'
        metadata = json.loads(metadata_path.read_text())
        metadata['base_fingerprint'] = 'wrong'
        metadata_path.write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError,'different base'):
            train_lora(toy.Manager(),self.fixture.manifest,self.root/'wrong-base',rank=2,alpha=4,
                       training_steps=1,options=self.opts,initial_adapter=self.source,log=lambda _:None)
        self.assertFalse((self.root/'wrong-base').exists())

    def test_compressed_targets_warm_start_and_failed_preflight_does_not_block_retry(self):
        config_path = self.source/'adapter_config.json'
        config = json.loads(config_path.read_text())
        config['target_modules'] = ['proj','stop_head']
        config_path.write_text(json.dumps(config))
        output = self.root/'retry'
        kwargs = dict(rank=2,alpha=4,training_steps=1,gradient_accumulation=1,
                      options=self.opts,initial_adapter=self.source,log=lambda _:None)
        with self.assertRaisesRegex(ValueError,'extra=.*stop_head'):
            train_lora(toy.Manager(),self.fixture.manifest,output,**kwargs)
        self.assertFalse(output.exists())
        config['target_modules'] = ['proj']
        config_path.write_text(json.dumps(config))
        train_lora(toy.Manager(),self.fixture.manifest,output,**kwargs)
        self.assertTrue((output/'trainer_state.pt').is_file())

    def test_legacy_schema3_default_options_still_resume(self):
        state_path = self.source/'trainer_state.pt'
        state = torch.load(state_path,weights_only=True)
        for name in ('sampling_strategy','validation_patches_per_example'):
            state['recipe']['options'].pop(name)
        torch.save(state,state_path)
        train_lora(toy.Manager(),self.fixture.manifest,self.source,rank=2,alpha=4,
                   training_steps=2,gradient_accumulation=1,save_every=1,options=self.opts,
                   resume_checkpoint=self.source,log=lambda _:None)
        restored = torch.load(state_path,weights_only=True)
        self.assertEqual(restored['step'],2)
        self.assertEqual(restored['recipe']['options']['sampling_strategy'],'speaker_balanced')

"""Coverage, all-patch objectives, and a new stage initialized from an adapter."""
from dataclasses import replace
import json
from pathlib import Path
import random
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import load_file

from fireredtts3_local.training_batch import Example, sample_flow
from fireredtts3_local.training_data import SpeakerPairs
from fireredtts3_local.training_run import (TrainingOptions, initial_adapter_info,
    planned_batches, train_lora)
import test_russian_training as toy


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

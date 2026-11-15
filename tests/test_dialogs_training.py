"""Coverage with replay and independent unseen-voice/domain validation."""
from dataclasses import asdict, replace
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf
from safetensors.torch import load_file
import torch

from fireredtts3_local.training_data import load_manifest, write_manifest
from fireredtts3_local.training_features import precompute_recipe
from fireredtts3_local.training_run import (TrainingOptions, planned_batches, prepare_run_data,
                                          primary_examples_per_update, train_lora)
import test_russian_training as toy


class DialogsTrainingTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        counter=0
        def make(name,speakers,count):
            nonlocal counter
            rows=[]
            for speaker in speakers:
                for i in range(count):
                    audio=self.root/f'{name}-{speaker}-{i}.wav'
                    sf.write(audio,np.sin(np.arange(24000)*(.02+counter*.001))*.1,24000)
                    counter+=1
                    rows.append(dict(audio=str(audio),text=f'Реплика {name} {speaker} {i}.', language='Russian',
                        speaker_id=speaker,source_id=f'{name}-{speaker}',utt_id=f'{name}-{speaker}-{i}'))
            return write_manifest(self.root/f'{name}.jsonl',rows)
        self.train=make('train','ab',5)
        self.domain=make('adaptation','ab',2)
        self.unseen=make('unseen','u',2)
        self.replay=make('replay','xyz',2)
        self.options=TrainingOptions(validation_manifest=str(self.unseen), adaptation_validation_manifest=str(self.domain),
            replay_manifest=str(self.replay),replay_ratio=.5,sampling_strategy='coverage_replay',
            batch_size=4,validation_batch_size=2,validation_pairs=4,validate_every='epoch',patience=0,
            max_prompt_seconds=10,max_target_seconds=10,patches_per_example='all',validation_patches_per_example=3,
            cache_dir=str(self.root/'cache'),gradient_checkpointing=False,dit_chunk_size=2,batch_token_budget=1)

    def test_each_primary_once_and_equal_replay_even_partial_update_and_resume(self):
        train,val,replay,adaptation=prepare_run_data(self.train,self.options,1234,include_adaptation_validation=True)
        self.assertEqual((len(train.rows),len(val.rows),len(replay.rows),len(adaptation.rows)),(10,2,6,4))
        stream=list(planned_batches(train,replay,self.options,random.Random(9).getstate(),0,5,1,'Epochs'))
        pairs=[pair for batch in stream for pair in batch.pairs]
        primary=[target['utt_id'] for _,target,_ in pairs if target['utt_id'].startswith('train-')]
        self.assertCountEqual(primary,[r['utt_id'] for r in train.rows])
        self.assertEqual(len(pairs),20)
        self.assertTrue(all(keep==.1 for _,_,keep in pairs))
        commits=[i for i,b in enumerate(stream) if b.sampler_state is not None]
        self.assertEqual(commits,[3,7,11,15,19])
        resumed=list(planned_batches(train,replay,self.options,stream[7].sampler_state,2,5,1,'Epochs'))
        self.assertEqual(resumed,stream[8:])
        # 10 primary clips, 3 primary + 3 replay per full update, then 1 + 1.
        opts=replace(self.options,batch_size=6)
        partial=list(planned_batches(train,replay,opts,random.Random(9).getstate(),0,4,1,'Epochs'))
        self.assertEqual(sum(len(b.pairs) for b in partial),20)
        self.assertEqual(sum(b.sampler_state is not None for b in partial),4)
        with self.assertRaisesRegex(ValueError,'integer replay'):
            primary_examples_per_update(replace(opts,replay_ratio=.2),1)

    def test_secondary_validation_rejects_audio_source_or_text_leakage(self):
        rows=load_manifest(self.domain)
        train=load_manifest(self.train)
        rows[0]['audio']=train[0]['audio']
        write_manifest(self.root/'bad.jsonl',rows)
        with self.assertRaisesRegex(ValueError,'leakage'):
            prepare_run_data(self.train,replace(self.options,adaptation_validation_manifest=str(self.root/'bad.jsonl')),1234)
        rows=load_manifest(self.domain)
        rows[0]['text']=train[0]['text'].replace('.', '?')
        write_manifest(self.root/'bad.jsonl',rows)
        with self.assertRaisesRegex(ValueError,'text leakage'):
            prepare_run_data(self.train,replace(self.options,adaptation_validation_manifest=str(self.root/'bad.jsonl')),1234)
        rows=load_manifest(self.domain)
        rows[0]['text']=load_manifest(self.replay)[0]['text']
        write_manifest(self.root/'bad.jsonl',rows)
        with self.assertRaisesRegex(ValueError,'text leakage'):
            prepare_run_data(self.train,replace(self.options,adaptation_validation_manifest=str(self.root/'bad.jsonl')),1234)

    def test_dialogs_only_coverage_evaluates_every_holdout_target_without_sampling(self):
        opts = replace(self.options, validation_manifest=str(self.domain), validation_scope='recording',
            adaptation_validation_manifest='', replay_manifest='', replay_ratio=0.,
            sampling_strategy='coverage', validation_pairs=1, validation_all_targets=True)
        with patch('fireredtts3_local.training_run.require_training_hardware'), \
             patch('fireredtts3_local.training_run.model_identity', return_value='toy'), \
             patch('fireredtts3_local.training_data.SpeakerPairs.sample',
                   side_effect=AssertionError('Full coverage/validation must not sample targets')):
            train_lora(toy.Manager(), self.train, self.root/'dialogs-only', rank=2, alpha=4,
                       length_mode='Epochs', training_epochs=2, gradient_accumulation=1,
                       save_every=1, options=opts, log=lambda _: None)
        scores = [json.loads(line) for line in (self.root/'dialogs-only/validation_metrics.jsonl').read_text().splitlines()]
        self.assertEqual([(r['step'], r['targets']) for r in scores], [(0,4), (3,4), (6,4)])
        coverage = [json.loads(line) for line in (self.root/'dialogs-only/coverage.jsonl').read_text().splitlines()]
        self.assertEqual([r['targets'] for r in coverage], [10,10])
        self.assertTrue(all(not r.get('replay_targets') for r in coverage))

    def test_precompute_both_validations_and_exact_resumed_training(self):
        from fireredtts3_local.training import STOP_EVENT
        opts=replace(self.options,require_precomputed_features=True,offload_frozen_encoders=True)
        recipe=dict(manifest_path=str(self.train),options=asdict(opts),seed=1234)
        (self.root/'recipe.json').write_text(json.dumps(recipe))
        torch.manual_seed(1234)
        pipeline=toy.tts(); pipeline.tts_core.patch_encoder.requires_grad_(False)
        with patch('fireredtts3_local.training_run.require_training_hardware'), \
             patch('fireredtts3_local.training_features.model_identity',return_value='toy'), \
             patch('fireredtts3_local.training_features.load_feature_pipeline',return_value=pipeline):
            report=precompute_recipe(self.root/'recipe.json','unused-base',log=lambda _:None)
        self.assertEqual(report['unique_audio'],22)  # train + replay + both validations
        kwargs=dict(rank=2,alpha=4,length_mode='Epochs',training_epochs=2,gradient_accumulation=1,
                    save_every=1,options=opts,log=lambda _:None)
        def stop(step,*_):
            if step==2: STOP_EVENT.set()
        with patch('fireredtts3_local.training_run.require_training_hardware'), \
             patch('fireredtts3_local.training_run.model_identity',return_value='toy'), \
             patch.object(toy.Encoder,'encode',side_effect=AssertionError('must use precomputed features')):
            train_lora(toy.Manager(),self.train,self.root/'paused',progress_cb=stop,**kwargs)
            train_lora(toy.Manager(),self.train,self.root/'paused',resume_checkpoint=self.root/'paused',**kwargs)
            train_lora(toy.Manager(),self.train,self.root/'full',**kwargs)
        a=load_file(str(self.root/'paused'/'adapter_model.safetensors'))
        b=load_file(str(self.root/'full'/'adapter_model.safetensors'))
        for name in a: torch.testing.assert_close(a[name],b[name],rtol=0,atol=0)
        coverage=[json.loads(line) for line in (self.root/'full'/'coverage.jsonl').read_text().splitlines()]
        self.assertEqual([(r['targets'],r['replay_targets']) for r in coverage],[(10,10),(10,10)])
        domain_scores=[json.loads(line) for line in (self.root/'full'/'adaptation_validation_metrics.jsonl').read_text().splitlines()]
        self.assertEqual([r['targets'] for r in domain_scores],[4,4,4])
        scores=[json.loads(line) for line in (self.root/'full'/'selection_metrics.jsonl').read_text().splitlines()]
        self.assertEqual([r['step'] for r in scores],[0,5,10])
        for row in scores:
            self.assertEqual(row['loss'],(row['unseen_voice_loss']+row['adaptation_loss'])/2)
        state=torch.load(self.root/'full'/'trainer_state.pt',weights_only=True)
        self.assertIn('adaptation_validation_fingerprint',state['recipe'])
        self.assertEqual(state['best_validation'],min(r['loss'] for r in scores))
        domain=load_manifest(self.domain)
        domain[0]['text']='Изменённая проверочная реплика.'
        write_manifest(self.domain,domain)
        with patch('fireredtts3_local.training_run.require_training_hardware'), self.assertRaisesRegex(ValueError,'Resume recipe/data differ'):
            train_lora(toy.Manager(),self.train,self.root/'full',resume_checkpoint=self.root/'full',**kwargs)

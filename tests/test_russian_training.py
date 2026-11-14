import copy
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
from torch import nn

from fireredtts3_local.training_data import (SpeakerPairs, assert_disjoint, load_manifest,
    plain_text, split_manifest, write_manifest)
from fireredtts3_local.training_objective import adapter_targets, conditioning, flow_loss
from fireredtts3_local.training_run import TrainingOptions, isolated_rng, prepare_run_data, train_lora
from fireredtts3_local.yo_dataset import aligned_segments, import_yo


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(32, 8)
        self.proj = nn.Linear(8, 8)
    def forward(self, inputs_embeds, use_cache=False, attention_mask=None, position_ids=None):
        return SimpleNamespace(last_hidden_state=self.proj(inputs_embeds).cumsum(1))


class PatchEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 8)
    def forward(self, x):
        return self.linear(x.reshape(1, -1, 4))


class Dit(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(13, 2)
        self.last_input = None
    def forward(self, x, t):
        self.last_input = x.detach()
        return self.linear(x)


class Core(nn.Module):
    patch_size = 2
    history_patches = 2
    history_length = 4
    def __init__(self):
        super().__init__()
        self.backbone_llm = Backbone()
        self.patch_encoder = PatchEncoder()
        self.spk_proj_llm = nn.Linear(3, 8)
        self.spk_proj_dit = nn.Linear(3, 3)
        self.dit_head = nn.Linear(8, 8)
        self.stop_head = nn.Linear(8, 1)
        self.dit = Dit()


class Encoder(nn.Module):
    sample_rate = 24000
    downsample_rate = 2400
    def pad_to_multiple_of(self, audio, n):
        return torch.nn.functional.pad(audio, (0, (-audio.shape[-1]) % n))
    def encode(self, audio, sr):
        x = audio.reshape(1, -1, self.downsample_rate).mean(-1)
        return torch.stack([x, x * 0.5], -1)


class Speaker(nn.Module):
    def forward(self, audio, sr):
        return torch.tensor([[0.1, 0.2, 0.3]])


def tts():
    return SimpleNamespace(tts_core=Core(), device=torch.device('cpu'), redae=Encoder(),
                           spk_extractor=Speaker(), _tokenize_text=lambda s: torch.tensor([[1, 2, 3]]))


class Manager:
    def load_base(self, **kwargs):
        self.tts = tts()
        return self.tts
    def unload(self, log=print):
        pass


class DataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.rows = []
        for speaker in range(4):
            for clip in range(3):
                path = self.root / f'{speaker}-{clip}.wav'
                sf.write(path, np.sin(np.arange(24000)*(.02+speaker*.01+clip*.001))*.1, 24000)
                self.rows.append(dict(audio=str(path), text='Привет, мир!', language='ru',
                                      speaker_id=str(speaker), source_id=f'recording-{speaker}'))
        self.manifest = write_manifest(self.root/'data.jsonl', self.rows)
    def test_manifest_requires_identity(self):
        self.rows[0].pop('speaker_id')
        write_manifest(self.manifest, self.rows)
        with self.assertRaisesRegex(ValueError, 'speaker_id'):
            load_manifest(self.manifest)
    def test_stress_and_arithmetic(self):
        self.assertEqual(plain_text('Прив+ет! 2 + 2. Ещё\u0301.'), 'Привет! 2 + 2. Ещё.')
    def test_split_and_same_speaker_pairs(self):
        rows = load_manifest(self.manifest)
        train, val = split_manifest(rows, .25, 1)
        self.assertEqual(len(val), 3)
        assert_disjoint(train, val)
        sampler = SpeakerPairs(train)
        rng = random.Random(4)
        for _ in range(50):
            a,b = sampler.sample(rng)
            self.assertEqual(a['speaker_id'], b['speaker_id'])
            self.assertNotEqual(a['audio_sha256'], b['audio_sha256'])
    def test_global_identity_connects_local_speakers(self):
        rows = load_manifest(self.manifest)
        for row in rows[:6]: row['global_spk_id'] = 'same-person'
        train, val = split_manifest(rows, .25, 5)
        self.assertFalse(any(r.get('global_spk_id') for r in train) and any(r.get('global_spk_id') for r in val))
    def test_duplicate_audio_leakage(self):
        rows = load_manifest(self.manifest)
        with self.assertRaisesRegex(ValueError, 'leakage'):
            assert_disjoint(rows[:1], [dict(rows[0], speaker_id='other')])
    def test_long_audio_not_truncated(self):
        rows = load_manifest(self.manifest)
        rows[0]['duration']=21
        with self.assertRaisesRegex(ValueError, 'segment'):
            SpeakerPairs(rows)
    def test_singleton_fails_before_model_loading(self):
        with self.assertRaisesRegex(ValueError, 'reference'):
            SpeakerPairs(load_manifest(self.manifest)[:1])
    def test_overlapping_references_rejected(self):
        rows=load_manifest(self.manifest)[:2]
        for row in rows: row.update(start=0, end=1)
        with self.assertRaisesRegex(ValueError, 'reference'):
            SpeakerPairs(rows)
    def test_separate_utterance_clocks_in_same_video(self):
        rows=load_manifest(self.manifest)[:2]
        for i,row in enumerate(rows):
            row.update(start=0,end=1,segment_source_id=f'utterance-{i}')
        self.assertTrue(SpeakerPairs.distinct(*rows))
        self.assertEqual(len(SpeakerPairs(rows).groups),1)
    def test_replay_validation_leakage(self):
        rows=load_manifest(self.manifest)
        write_manifest(self.root/'train.jsonl', rows[3:])
        write_manifest(self.root/'val.jsonl', rows[:3])
        opts=TrainingOptions(validation_manifest=str(self.root/'val.jsonl'), replay_manifest=str(self.root/'val.jsonl'))
        with self.assertRaisesRegex(ValueError, 'leakage'):
            prepare_run_data(self.root/'train.jsonl', opts, 1)
    def test_cpu_training_fails_after_data_checks(self):
        with patch('torch.cuda.is_available', return_value=False):
            with self.assertRaisesRegex(RuntimeError, 'Dataset validation passed'):
                train_lora(Manager(), self.manifest, self.root/'run', log=lambda _:None)
    def test_training_resume_matches_uninterrupted(self):
        from fireredtts3_local.training import STOP_EVENT
        opts=TrainingOptions(validation_pairs=2, validate_every=1, patience=0,
                             gradient_checkpointing=False, patches_per_example=3, cache_dir="")
        kwargs=dict(rank=2, alpha=4, training_steps=2, gradient_accumulation=2,
                    save_every=1, options=opts, log=lambda _:None)
        with patch('fireredtts3_local.training_run.require_training_hardware'), patch('fireredtts3_local.training_run.model_identity', return_value='toy-model'):
            train_lora(Manager(), self.manifest, self.root/'full', **kwargs)
            train_lora(Manager(), self.manifest, self.root/'resume', progress_cb=lambda *a:STOP_EVENT.set(), **kwargs)
            train_lora(Manager(), self.manifest, self.root/'resume', resume_checkpoint=self.root/'resume', **kwargs)
        from safetensors.torch import load_file
        a=load_file(str(self.root/'full'/'adapter_model.safetensors'))
        b=load_file(str(self.root/'resume'/'adapter_model.safetensors'))
        for key in a: torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
        self.assertTrue((self.root/'full'/'best-validation'/'adapter_config.json').is_file())


class ObjectiveTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)
        self.tts=tts()
        self.prompt=torch.randn(1,4,2)
        self.target=torch.randn(1,6,2)
        self.speaker=torch.randn(1,3)
    def test_terminal_state_is_after_all_target_patches(self):
        core=self.tts.tts_core
        cond,spk,logits=conditioning(core,torch.tensor([[1,2]]),self.prompt,self.target,self.speaker)
        self.assertEqual(logits.shape,(4,)) # three patches to emit, then STOP
        inputs=torch.cat([core.spk_proj_llm(self.speaker)[:,None],core.backbone_llm.embed_tokens(torch.tensor([[1,2]])),core.patch_encoder(self.prompt),core.patch_encoder(self.target)],1)
        expected=core.stop_head(core.backbone_llm(inputs).last_hidden_state[:,-1]).flatten()
        torch.testing.assert_close(logits[-1:], expected)
    def test_stop_targets_and_flow_gradients(self):
        real=torch.nn.functional.binary_cross_entropy_with_logits
        captured=[]
        def record(logits, targets):
            captured.append(targets.detach().tolist())
            return real(logits,targets)
        with patch('fireredtts3_local.training_objective.F.binary_cross_entropy_with_logits',side_effect=record):
            loss, metrics=flow_loss(self.tts,self.prompt,self.speaker,self.target,'А.','Б.','Russian',3,0)
        self.assertEqual(captured, [[0.,0.,0.,1.]])
        loss.backward()
        self.assertGreater(self.tts.tts_core.backbone_llm.proj.weight.grad.abs().sum(),0)
        self.assertGreater(self.tts.tts_core.dit.linear.weight.grad.abs().sum(),0)
    def test_cfg_drops_conditions_but_preserves_history(self):
        flow_loss(self.tts,self.prompt,self.speaker,self.target,'А.','Б.','Russian',3,1)
        x=self.tts.tts_core.dit.last_input
        self.assertEqual(x[:,:,2:].abs().sum(),0)
        self.assertGreater(x[:,:4,:2].abs().sum(),0)
    def test_profiles_exclude_identity_and_codec(self):
        core=self.tts.tts_core
        self.assertEqual(adapter_targets(core,'backbone'),['backbone_llm.proj'])
        names=adapter_targets(core,'backbone_dit')
        self.assertIn('dit.linear', names)
        self.assertNotIn('spk_proj_dit', names)
        self.assertNotIn('stop_head', names)
        self.assertNotIn('patch_encoder.linear', names)
    def test_peft_teacher_restores_mode_and_frozen_base(self):
        from peft import LoraConfig, get_peft_model
        core=get_peft_model(self.tts.tts_core,LoraConfig(r=2,lora_alpha=4,target_modules=adapter_targets(self.tts.tts_core,'backbone')))
        self.tts.tts_core=core
        core.train()
        loss, metrics=flow_loss(self.tts,self.prompt,self.speaker,self.target,'А.','Б.','Russian',3,0,preservation_weight=.1)
        loss.backward()
        self.assertTrue(core.training)
        self.assertTrue(all(p.grad is None for n,p in core.named_parameters() if 'lora_' not in n))
        self.assertTrue(any(p.grad is not None for n,p in core.named_parameters() if 'lora_' in n))
    def test_validation_rng_isolation(self):
        torch.manual_seed(13)
        state=torch.get_rng_state()
        python_state=random.getstate()
        with isolated_rng(100):
            torch.randn(3);random.random()
        self.assertTrue(torch.equal(state,torch.get_rng_state()))
        self.assertEqual(python_state,random.getstate())


class ImportTests(unittest.TestCase):
    def row(self):
        return dict(text='привет мир это тест',text_denorm='Прив+ет, мир. Это тест!',
                    text_alignment='прив[+]ет(0.10,0.80) мир(0.90,1.80) [pause0.3](1.80,2.10) это(2.10,2.80) тест(2.90,3.80)',duration=4.)
    def test_segment_text_matches_audio_bounds(self):
        segments=aligned_segments(self.row(),max_seconds=2)
        self.assertEqual([x[2] for x in segments],['Привет, мир.','Это тест!'])
        self.assertLessEqual(segments[0][1],segments[1][0])
    def test_rejects_changed_denormalized_content(self):
        row=self.row();row['text_denorm']='Привет мир это 12.'
        with self.assertRaisesRegex(ValueError,'denormalized'):
            aligned_segments(row)
    def test_rejects_inconsistent_alignment(self):
        row=self.row();row['text']='привет мир другое'
        with self.assertRaisesRegex(ValueError,'mismatch'):
            aligned_segments(row)
    def test_jsonl_import_real_audio(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            sf.write(root/'clip.wav',np.sin(np.arange(96000)*.1)*.1,24000)
            row=self.row();row.update(audio='clip.wav',utt_id='x',local_spk_id='video__spk0',global_spk_id=None,lang='ru',mos_score=4.5)
            (root/'source.jsonl').write_text(json.dumps(row),encoding='utf-8')
            manifest=import_yo([root/'source.jsonl'],root/'out',max_seconds=2,source_revision='test',log=lambda _:None)
            rows=load_manifest(manifest)
            self.assertEqual(len(rows),2)
            pairs=SpeakerPairs(rows)
            self.assertEqual(len(pairs.groups),1)


if __name__=='__main__': unittest.main()

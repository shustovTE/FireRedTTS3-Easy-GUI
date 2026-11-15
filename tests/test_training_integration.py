import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf
import torch

from fireredtts3_local.evaluation import compare_reports, word_error_rate
from fireredtts3_local.training_objective import adapter_targets, conditioning, flow_loss, FeatureCache
from fireredtts3_local.training_run import TrainingOptions
from fireredtts3_local.yo_dataset import import_yo
from test_russian_training import tts


class UpstreamTests(unittest.TestCase):
    def test_real_qwen_patch_encoder_dit_backward_and_incremental_states(self):
        from fireredtts3_local.model_manager import ModelManager
        ModelManager._ensure_upstream_namespace()
        import fireredtts3.llm.fireredtts3_base as upstream
        cfg=dict(upstream.Qwen3_1_7B_ConfigDict)
        cfg.update(hidden_size=32,intermediate_size=64,num_hidden_layers=2,num_attention_heads=4,
                   num_key_value_heads=2,head_dim=8,vocab_size=64,attn_implementation='sdpa')
        with patch.object(upstream,'Qwen3_1_7B_ConfigDict',cfg):
            core=upstream.FireRedTTS3BaseCore(upstream.FireRedTTS3BaseCoreConfig(
                redae_dim=4,patch_size=2,spk_in_dim=6,patch_encoder_hidden_size=16,
                patch_encoder_depth=1,patch_encoder_num_heads=2,dit_hidden_size=16,dit_depth=1,dit_num_heads=2))
        core.eval()
        prompt,target,speaker=torch.randn(1,4,4),torch.randn(1,6,4),torch.randn(1,6)
        tokens=torch.tensor([[1,2,3]])
        conditions, _, logits=conditioning(core,tokens,prompt,target,speaker)
        # Independent autoregressive rollout must match the teacher-forced stop states.
        inputs=torch.cat([core.spk_proj_llm(speaker)[:,None],core.backbone_llm.embed_tokens(tokens),core.patch_encoder(prompt)],1)
        cache=None
        states=[]
        history=torch.zeros(1,core.history_patches,32)
        for j in range(4):
            result=core.backbone_llm(inputs_embeds=inputs,use_cache=True,past_key_values=cache)
            cache=result.past_key_values
            states.append(core.stop_head(result.last_hidden_state[:,-1]).flatten())
            new_hidden = result.last_hidden_state[:,-2:] if j==0 else result.last_hidden_state[:,-1:]
            history=torch.cat([history,new_hidden],1)
            if j<3:
                expected_condition=core.dit_head(history[:,-core.history_patches-1:])
                torch.testing.assert_close(conditions[j:j+1],expected_condition,rtol=1e-4,atol=1e-5)
            if j<3:inputs=core.patch_encoder(target[:,j*2:j*2+2])
        torch.testing.assert_close(logits,torch.cat(states),rtol=1e-4,atol=1e-5)
        from peft import LoraConfig,get_peft_model
        core=get_peft_model(core,LoraConfig(r=2,lora_alpha=4,target_modules=adapter_targets(core,'backbone_dit')))
        core.backbone_llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        core.train()
        pipeline=SimpleNamespace(tts_core=core,_tokenize_text=lambda _:tokens)
        loss,_=flow_loss(pipeline,prompt,speaker,target,'Привет.','Как дела?','Russian',3,.1,preservation_weight=.1)
        loss.backward()
        for prefix in ('backbone_llm','dit.'):
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for n,p in core.named_parameters() if prefix in n and 'lora_' in n))

    def test_cache_reuses_features_and_invalidates_model_identity(self):
        pipeline=tts()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            sf.write(root/'audio.wav',np.sin(np.arange(24000)*.1)*.1,24000)
            from fireredtts3_local.training_data import file_digest
            row={'audio':str(root/'audio.wav'),'audio_sha256':file_digest(root/'audio.wav')}
            cache=FeatureCache(pipeline,root/'cache','model-one')
            a=cache.get(row)
            with patch.object(pipeline.redae,'encode',side_effect=AssertionError('should use cache')):
                b=cache.get(row)
            for x,y in zip(a,b):torch.testing.assert_close(x,y)
            with patch.object(pipeline.redae,'encode',side_effect=RuntimeError('new model')):
                with self.assertRaisesRegex(RuntimeError,'new model'):
                    FeatureCache(pipeline,root/'cache','model-two').get(row)


class EvaluationTests(unittest.TestCase):
    def test_wer(self):
        self.assertEqual(word_error_rate('Прив+ет, мир!','привет мир'),0)
        self.assertEqual(word_error_rate('один два три','один три'),1/3)
    def test_quality_gate_checks_worst_speaker_not_just_mean(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            def report(scores):
                return {'benchmark_sha256':'same','speaker_model':'independent','asr_model':'asr','complete':True,
                        'results':[{'id':str(i),'speaker_id':str(i),'speaker_similarity':score,'wer':.1} for i,score in enumerate(scores)]}
            (root/'a').write_text(json.dumps(report([.8,.8])))
            (root/'b').write_text(json.dumps(report([.9,.7])))
            self.assertFalse(compare_reports(root/'a',root/'b')['automatic_gate_passed'])
            (root/'b').write_text(json.dumps(report([.81,.8])))
            self.assertTrue(compare_reports(root/'a',root/'b')['automatic_gate_passed'])
    def test_nan_options_rejected(self):
        with self.assertRaises(ValueError):TrainingOptions.parse({'preservation_weight':float('nan')})
    def test_unknown_options_rejected(self):
        with self.assertRaises(TypeError):TrainingOptions.parse({'typo':1})
    def test_real_parquet_import(self):
        import io
        import pyarrow as pa
        import pyarrow.parquet as pq
        from test_russian_training import ImportTests
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            buf=io.BytesIO()
            sf.write(buf,np.sin(np.arange(96000)*.1)*.1,24000,format='WAV')
            row=ImportTests().row()
            row.update(audio={'bytes':buf.getvalue(),'path':'sample.wav'},utt_id='x',local_spk_id='video__spk0',lang='ru',mos_score=4.5)
            pq.write_table(pa.Table.from_pylist([row]),root/'data.parquet')
            manifest=import_yo([root/'data.parquet'],root/'result',max_seconds=2,source_revision='revision',log=lambda _:None)
            self.assertEqual(len(manifest.read_text().splitlines()),2)


if __name__=='__main__':unittest.main()

class ExportTests(unittest.TestCase):
    def test_complete_core_merge_and_export_keys(self):
        from fireredtts3_local.model_manager import ModelManager
        ModelManager._ensure_upstream_namespace()
        import fireredtts3.llm.fireredtts3_base as upstream
        from fireredtts3.redae.redae import RedAEConfig
        from fireredtts3_local.export import export_merged
        from fireredtts3_local.training_data import model_identity
        from peft import LoraConfig, get_peft_model
        from safetensors.torch import load_file
        cfg=dict(upstream.Qwen3_1_7B_ConfigDict)
        cfg.update(hidden_size=32,intermediate_size=64,num_hidden_layers=1,num_attention_heads=4,
                   num_key_value_heads=2,head_dim=8,vocab_size=64,attn_implementation='sdpa')
        with tempfile.TemporaryDirectory() as directory, patch.object(upstream,'Qwen3_1_7B_ConfigDict',cfg):
            root=Path(directory);base=root/'base'
            core=upstream.FireRedTTS3BaseCore(upstream.FireRedTTS3BaseCoreConfig(
                redae_dim=4,patch_size=2,spk_in_dim=6,patch_encoder_hidden_size=16,
                patch_encoder_depth=1,patch_encoder_num_heads=2,dit_hidden_size=16,dit_depth=1,dit_num_heads=2))
            core.save_pretrained(base/'fireredtts3_base')
            redae=RedAEConfig(bottleneck_dim=4);redae.save_pretrained(base/'redae')
            (base/'campp').mkdir();(base/'campp'/'campplus_voxceleb.bin').write_bytes(b'fixture')
            (base/'text_tokenizer').mkdir();(base/'text_tokenizer'/'tokenizer_config.json').write_text('{}')
            template={**redae.to_dict(),**core.config.to_dict(),**core.backbone_llm.config.to_dict(),'model_type':'fireredtts3', 'rope_theta':cfg['rope_theta']}
            (root/'config.json').write_text(json.dumps(template))
            original={key:value.clone() for key,value in core.state_dict().items()}
            core=get_peft_model(core,LoraConfig(r=2,lora_alpha=4,target_modules=adapter_targets(core,'backbone_dit')))
            for name,parameter in core.named_parameters():
                if 'lora_B' in name:parameter.data.fill_(.1)
            core.save_pretrained(root/'adapter')
            (root/'adapter'/'firered_lora_metadata.json').write_text(json.dumps({
                'base_fingerprint':model_identity(base), 'recipe':{'options':{'profile':'backbone_dit'}}}))
            export_merged(base,root/'adapter',root/'config.json',root/'export')
            exported=load_file(str(root/'export'/'model.safetensors'))
            self.assertEqual(exported.keys(),original.keys())
            self.assertFalse(torch.equal(exported['dit_head.weight'],original['dit_head.weight']))
            torch.testing.assert_close(exported['spk_proj_dit.weight'],original['spk_proj_dit.weight'])
            self.assertTrue((root/'export'/'redae'/'config.json').is_file())
            with self.assertRaisesRegex(ValueError,'must not exist'):
                export_merged(base,root/'adapter',root/'config.json',root/'export')

class GuiTests(unittest.TestCase):
    def test_recipe_and_speaker_survive_project_roundtrip(self):
        import os
        os.environ['GRADIO_ANALYTICS_ENABLED']='False'
        os.environ['XDG_CACHE_HOME']=str(Path('.runtime/test-cache').resolve())
        import easy_gui as gui
        from dataclasses import asdict
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            reference=root/'ref.wav';reference.write_bytes(b'copy-fixture')
            recipe=json.dumps(asdict(TrainingOptions(profile='backbone_dit')))
            with patch.object(gui,'TRAINING_PROJECTS',root/'projects'):
                gui.save_training_project('roundtrip','source','Russian','manifest.jsonl','adapter',
                    '24 GB',16,32,.05,1e-5,'Steps',100,2,8,1234,10,1,False,'Тест.',str(reference),
                    'Референс.','Russian','model','Russian',8,recipe,'speaker-1')
                loaded=gui.load_training_project('roundtrip')
                self.assertEqual(len(loaded),26)
                self.assertEqual(json.loads(loaded[-2]['value'])['profile'],'backbone_dit')
                self.assertEqual(loaded[-1]['value'],'speaker-1')
                self.assertEqual(Path(loaded[18]['value']).read_bytes(),b'copy-fixture')
            configs=gui.demo.get_config_file()['dependencies']
            training=next(d for d in configs if d.get('api_name')=='run_lora_training')
            self.assertEqual(len(training['inputs']),21)

    def test_autotune_reads_relative_import_paths(self):
        from fireredtts3_local.projects import analyze_manifest,autotune
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            sf.write(root/'audio.wav',np.zeros(24000),24000)
            (root/'data.jsonl').write_text(json.dumps({'audio':'audio.wav','speaker_id':'a'})+'\n')
            stats=analyze_manifest(root/'data.jsonl')
            self.assertEqual(stats['sample_count'],1)
            self.assertEqual(stats['speaker_count'],1)
            self.assertEqual(autotune(root/'data.jsonl','24 GB')['suggested_steps'],50)

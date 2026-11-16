"""CPU native-core rollouts: tracing parity, provenance, matrix isolation/resume."""
import copy
import json
import os
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import soundfile as sf
import torch

from fireredtts3_local import cfg_diagnostics as diag
from fireredtts3_local.model_manager import ModelManager
from fireredtts3_local.training_data import file_digest, model_identity
from fireredtts3_local.training_job import read_status
from fireredtts3_local.training_monitor import SETTINGS, generate_case, load_benchmark
from test_hybrid_training import hybrid
from test_training_batch import tiny_core


class TinyPipeline:
    """Actual upstream AR loop/CFG/DiT; deterministic lightweight fake codec."""
    def __init__(self, core):
        self.tts_core=core
        self.redae=torch.nn.Identity()
        self.spk_extractor=torch.nn.Identity()
        self.device=torch.device('cpu')

    def generate(self, *, seed, inference_cfg, stop_threshold, n_timesteps, **_):
        torch.manual_seed(seed)
        output=self.tts_core.generate(spk_emb=torch.ones(1,6),text_tokens=torch.tensor([[1,2,3]]),
            prompt_latents=torch.ones(1,4,4)*.1,n_timesteps=n_timesteps,
            inference_cfg=inference_cfg,stop_threshold=stop_threshold,max_gen_steps=9)
        # Two frames per patch => 160 samples at 1 kHz. No waveform model download.
        return output[:,4:,0].repeat_interleave(80,dim=1)*.01,1000


class FakeManager:
    _attach_adapter=staticmethod(ModelManager._attach_adapter)
    def __init__(self, original):
        self.original=original
        self.base=None
        self.loads=0
    def load_base(self, **kwargs):
        if self.base is not None:raise AssertionError('Previous GPU stage was not released')
        self.base=TinyPipeline(copy.deepcopy(self.original).eval())
        self.loads+=1
        return self.base
    def unload(self, log=print):
        self.base=None


class DiagnosticTests(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup)
        self.root=Path(temporary.name)
        self.run=self.root/'run';self.run.mkdir()
        self.base=self.root/'base'
        for component in ('fireredtts3_base','redae','campp','text_tokenizer'):
            folder=self.base/component;folder.mkdir(parents=True)
            (folder/'config.json').write_text('{}')
        self.base_hash=model_identity(self.base)
        context=patch('fireredtts3_local.paths.MODEL_DIR',self.base)
        context.start();self.addCleanup(context.stop)
        context=patch.object(diag,'require_cuda')
        context.start();self.addCleanup(context.stop)
        threads=torch.get_num_threads();torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads,threads)
        torch.manual_seed(456)
        self.original=tiny_core()
        with torch.no_grad():
            self.original.stop_head.weight.zero_();self.original.stop_head.bias.fill_(3.)
        self.reference=self.root/'reference.wav'
        sf.write(self.reference,np.sin(np.arange(2400)*.1)*.1,24000)
        self.benchmark=self.root/'benchmark.jsonl'
        self.benchmark.write_text(''.join(json.dumps(dict(id=name,seed=1235,reference_audio=str(self.reference),
            reference_text='Тестовый голос.',text='Разные вопросы?' if i else 'Длинная реплика.',
            language='Russian',speaker_id='test'))+'\n' for i,name in enumerate(diag.DEFAULT_CASES)))
        rows,fingerprint=load_benchmark(self.benchmark)
        self.rows=rows
        recipe=dict(rank=2,alpha=4,dropout=0.,generation_benchmark_fingerprint=fingerprint,
                    options=dict(profile='hybrid',expected_base_fingerprint=self.base_hash,
                                 generation_benchmark=str(self.benchmark)))
        self.recipe=recipe
        (self.run/'recipe.json').write_text(json.dumps(recipe))
        base_results=[]
        for row in rows:
            _,_,metrics=generate_case(TinyPipeline(copy.deepcopy(self.original).eval()),row,SETTINGS)
            base_results.append(dict(id=row['id'],**metrics))
        previous=self.run/'step0.json'
        previous.write_text(json.dumps(dict(step=0,complete=True,fingerprint=fingerprint,settings=SETTINGS,results=base_results)))
        (self.run/'generation_metrics.jsonl').write_text(json.dumps(dict(step=0,report=str(previous)))+'\n')
        for step in (50,143,286):
            model=hybrid(copy.deepcopy(self.original))
            with torch.no_grad():
                for name,p in model.named_parameters():
                    if p.requires_grad and 'stop_head' not in name:p.add_(step/1_000_000)
            folder=self.run/(f'adapter-step-{step:06d}' if step==50 else f'checkpoint-step-{step:06d}')
            model.save_pretrained(folder)
            (folder/'firered_lora_metadata.json').write_text(json.dumps(dict(step=step,recipe=recipe,base_fingerprint=self.base_hash)))
        self.output=self.root/'diagnostics'

    def plan(self, output=None):
        return diag.make_plan(self.run,output or self.output)

    def test_trace_observes_native_cfg_without_changing_audio_weights_or_rng(self):
        pipeline=TinyPipeline(copy.deepcopy(self.original).eval())
        before={k:v.clone() for k,v in pipeline.tts_core.state_dict().items()}
        for cfg in (0.,1.,2.):
            settings={**SETTINGS,'inference_cfg':cfg}
            state=torch.get_rng_state()
            plain,rate,_=generate_case(pipeline,self.rows[0],settings)
            observed,observed_rate,result=generate_case(pipeline,self.rows[0],settings,trace_flow=True)
            np.testing.assert_array_equal(plain,observed)
            self.assertEqual(rate,observed_rate)
            torch.testing.assert_close(state,torch.get_rng_state(),rtol=0,atol=0)
            self.assertEqual(result['generated_patches'],6)
            self.assertEqual(len(result['flow_by_patch']),6)
            self.assertEqual(len(result['latent_levels']),7)
            self.assertEqual(result['finish_reason'],'stop')
            self.assertEqual(result['flow_by_patch'][0]['unconditional_rms'] is None,cfg==0)
        for name,value in pipeline.tts_core.state_dict().items():
            torch.testing.assert_close(value,before[name],rtol=0,atol=0)
        for module in (pipeline.tts_core.patch_encoder,pipeline.tts_core.stop_head,pipeline.tts_core.dit):
            self.assertFalse(module._forward_hooks)
            self.assertFalse(module._forward_pre_hooks)
        with patch.object(pipeline,'generate',side_effect=RuntimeError('synthetic failure')):
            with self.assertRaisesRegex(RuntimeError,'synthetic failure'):
                generate_case(pipeline,self.rows[0],trace_flow=True)
        self.assertFalse(pipeline.tts_core.dit._forward_hooks)
        self.assertFalse(pipeline.tts_core.stop_head._forward_hooks)

    def test_complete_matrix_and_resume_preserve_files_and_identical_generation(self):
        plan=self.plan()
        self.assertEqual(plan['total_cases'],24)
        self.assertEqual([s['step'] for s in plan['stages']],[0,50,143,286])
        originals={p:file_digest(p) for p in self.run.rglob('*') if p.is_file()}
        manager=FakeManager(self.original)
        full=diag.execute(plan,manager=manager,log=lambda _:None)
        self.assertTrue(full['complete']);self.assertEqual(manager.loads,4)
        for row in full['results']:
            self.assertEqual(row['seed'],1235)
            self.assertEqual(row['settings']['stop_threshold'],.5)
            self.assertFalse(row['settings']['do_split'])
            self.assertIsNotNone(row['rms_drift_delta_vs_base_same_cfg'])
            if row['stage']=='base':
                self.assertEqual(row['rms_drift_delta_vs_base_same_cfg'],0)
                if row['cfg']==2:self.assertFalse(row['previous_monitor']['reproduction_review_required'])
        for path,digest in originals.items():self.assertEqual(file_digest(path),digest)
        # A failed process leaves completed cases reusable, and a fresh process
        # restores the same weights/seeds for remaining cases.
        resumed_output=self.root/'resumed'
        resume_plan=self.plan(resumed_output)
        count=0
        def interrupt(*args,**kwargs):
            nonlocal count
            count+=1
            if count==3:raise RuntimeError('interrupted generation')
            return generate_case(*args,**kwargs)
        with patch.object(diag,'generate_case',side_effect=interrupt):
            with self.assertRaisesRegex(RuntimeError,'interrupted generation'):
                diag.execute(resume_plan,manager=FakeManager(self.original),log=lambda _:None)
        partial=json.loads((resumed_output/'report.json').read_text())
        self.assertEqual(partial['completed_cases'],2)
        keep={name:file_digest(resumed_output/name) for r in partial['results'] for name in r['artifacts']}
        resumed=diag.execute(resume_plan,resume=True,manager=FakeManager(self.original),log=lambda _:None)
        self.assertEqual([r['key'] for r in resumed['results']],[r['key'] for r in full['results']])
        for a,b in zip(full['results'],resumed['results']):
            # libsndfile's FLOAT WAV PEAK chunk includes a creation timestamp;
            # compare actual samples, not that container metadata across runs.
            left,_=sf.read(self.output/a['float_audio'])
            right,_=sf.read(resumed_output/b['float_audio'])
            np.testing.assert_array_equal(left,right)
        for name,digest in keep.items():self.assertEqual(file_digest(resumed_output/name),digest)
        # Result packages exclude original voice audio, full reference text,
        # raw floats (retained locally), adapters and the private source plan.
        for archive in self.output.glob('diagnostics-*.zip'):
            self.assertLess(archive.stat().st_size,32*1024*1024)
            with zipfile.ZipFile(archive) as bundle:
                self.assertTrue(all('reference' not in name and 'plan.json' not in name
                                    and 'safetensors' not in name and '.float.wav' not in name for name in bundle.namelist()))
        completed=self.output/full['results'][0]['audio']
        completed.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'audio changed'):
            diag.execute(plan,resume=True,manager=FakeManager(self.original),log=lambda _:None)

    def test_preflight_rejects_wrong_checkpoint_and_changed_inputs(self):
        for kwargs in (dict(steps=[51]),dict(steps=[0]),dict(cfgs=[float('nan')]),
                       dict(cfgs=[1,1]),dict(cases=['missing']),dict(output=self.run/'nested')):
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):
                diag.make_plan(self.run,**{'output':self.output,**kwargs})
        plan=self.plan()
        metadata=self.run/'adapter-step-000050/firered_lora_metadata.json'
        saved=metadata.read_bytes()
        data=json.loads(saved);data['step']=0;metadata.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError,'Checkpoint step/recipe'):self.plan()
        metadata.write_bytes(saved)
        weights=self.run/'adapter-step-000050/adapter_model.safetensors'
        weights.write_bytes(weights.read_bytes()+b'changed')
        with self.assertRaisesRegex(ValueError,'runtime changed'):diag.verify_plan(plan)
        sf.write(self.reference,np.zeros(2400),24000)
        with self.assertRaisesRegex(ValueError,'Benchmark text/reference'):self.plan()

    def test_relative_comparisons_use_base_at_same_cfg_not_cfg_two(self):
        plan=self.plan()
        def row(stage,cfg,drift):
            return dict(stage=stage,step=0 if stage=='base' else 50,cfg=cfg,id=self.rows[0]['id'],
                        finish_reason='stop',seconds=1.,end_minus_start_rms_db=drift,fraction_at_or_above_full_scale=0.)
        rows=[row('base',0.,10.),row('base',2.,0.),row('step-000050',0.,11.),row('step-000050',2.,4.)]
        diag.annotate(rows,plan)
        self.assertEqual(rows[2]['rms_drift_delta_vs_base_same_cfg'],1.)
        self.assertEqual(rows[2]['failures'],[])
        self.assertEqual(rows[3]['rms_drift_delta_vs_base_same_cfg'],4.)
        self.assertIn('level_drift_above_baseline_plus_3db',rows[3]['failures'])

    @unittest.skipUnless(os.name=='posix','POSIX detached jobs')
    def test_detached_launcher_holds_training_lock_and_records_child_failure(self):
        job=self.root/'job'
        result=diag.launch(self.run,self.output,job)
        self.assertEqual(os.getsid(result['pid']),result['pid'])
        with self.assertRaisesRegex(ValueError,'already owns'):
            diag.launch(self.run,self.root/'other-output',self.root/'other-job')
        deadline=time.monotonic()+30
        while time.monotonic()<deadline:
            state=read_status(job)
            if not state['active']:break
            time.sleep(.05)
        self.assertFalse(state['active'])
        # Child has neither test patches nor GPU/production base; must report
        # actual failure, not an invented successful diagnostic.
        self.assertEqual(state['status'],'failed')
        self.assertNotEqual(state['exit_code'],0)
        self.assertTrue((job/'job.log').stat().st_size)


if __name__=='__main__':unittest.main()

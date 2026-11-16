"""Read-only native check of the exact current pilot step, not best-validation."""
import copy
import json
from unittest.mock import patch
import unittest

from peft import LoraConfig
import torch

from fireredtts3_local import cfg_diagnostics as diag
from fireredtts3_local import rollout_training
from fireredtts3_local.history_probe import levels
from fireredtts3_local.hybrid import create_adapter
from fireredtts3_local.training_data import file_digest
from fireredtts3_local.training_objective import adapter_targets
from fireredtts3_local.training_monitor import SETTINGS, generate_case
import test_cfg_diagnostics as fixtures
from test_cfg_diagnostics import FakeManager, TinyPipeline


class RolloutCheckTests(unittest.TestCase):
    def setUp(self):
        fixture = fixtures.DiagnosticTests(); fixture.setUp(); self.addCleanup(fixture.doCleanups)
        self.fixture = fixture
        fixture.recipe['options'].update(profile='backbone', rollout={})
        (fixture.run/'recipe.json').write_text(json.dumps(fixture.recipe))
        self.models = {}
        for step in (143,240):
            model = copy.deepcopy(fixture.original)
            model = create_adapter(model, LoraConfig(r=2,lora_alpha=4,lora_dropout=0.,
                target_modules=adapter_targets(model,'backbone'))).eval()
            with torch.no_grad():
                for n,p in model.named_parameters():
                    if 'lora_B' in n:p.add_(step/10000)
            folder = fixture.run/f'checkpoint-step-{step:06d}' if step == 143 else fixture.run
            model.save_pretrained(folder)
            (folder/'firered_lora_metadata.json').write_text(json.dumps(dict(
                step=step,recipe=fixture.recipe,base_fingerprint=fixture.base_hash)))
            self.models[step] = model
        summaries = []
        for step in (0,143):
            results = []
            for i,row in enumerate(fixture.rows):
                model = self.models[step] if step else fixture.original
                samples, rate, metrics = generate_case(TinyPipeline(copy.deepcopy(model).eval()),row,SETTINGS)
                measured = levels(samples,rate)
                metrics.update({key:measured[key] for key in ('end_minus_start_rms_db','rms_window_seconds')})
                results.append(dict(id=row['id'], seed=row['seed'], audio=f'case-{i+6:05d}.wav', **metrics))
            report = dict(step=step, complete=True, fingerprint=fixture.recipe['generation_benchmark_fingerprint'],
                          settings=SETTINGS, results=results, nonoverlapping_windows=True)
            path = fixture.run/'generation_baseline.json' if step==0 else fixture.run/'step143.json'
            path.write_text(json.dumps(report))
            summaries.append(dict(step=step, report=str(path)))
        (fixture.run/'generation_metrics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in summaries))

    def test_latest_root_checkpoint_comparison_preserves_training_files_and_reproduces_controls(self):
        f = self.fixture
        plan = diag.make_plan(f.run,f.output,steps=[143,240],cfgs=[2.],cases=[f.rows[0]['id']])
        self.assertEqual(plan['adapter_profile'],'backbone')
        self.assertEqual(plan['stages'][-1]['adapter'],str(f.run))
        self.assertEqual(plan['stages'][-1]['step'],240)
        self.assertTrue(plan['nonoverlapping_windows'])
        before = {p:file_digest(p) for p in f.run.rglob('*') if p.is_file()}
        result = diag.execute(plan,manager=FakeManager(f.original),log=lambda _:None)
        self.assertTrue(result['complete']);self.assertEqual(result['completed_cases'],3)
        self.assertEqual([r['step'] for r in result['results']],[0,143,240])
        for row in result['results']:
            self.assertFalse(row['settings']['do_split'])
            self.assertEqual(row['seed'],f.rows[0]['seed'])
            self.assertIn('rms_window_seconds',row)
            if row['step'] in (0,143):
                self.assertFalse(row['previous_monitor']['reproduction_review_required'])
            else:
                self.assertIsNone(row['previous_monitor'])
        for path,digest in before.items():self.assertEqual(file_digest(path),digest)
        # Cannot relabel the root checkpoint as a requested missing step.
        with self.assertRaisesRegex(ValueError,'Missing saved checkpoint'):
            diag.make_plan(f.run,f.root/'bad',steps=[239],cfgs=[2.],cases=[f.rows[0]['id']])
        metadata=f.run/'firered_lora_metadata.json'
        saved=json.loads(metadata.read_text());saved['step']=241;metadata.write_text(json.dumps(saved))
        with self.assertRaisesRegex(ValueError,'Missing saved checkpoint'):
            diag.verify_plan(plan)

    def test_check_maps_monitor_filename_without_guessing_case_or_seed(self):
        f=self.fixture
        with patch.object(diag,'launch',return_value={'cases':3}) as launch:
            result=rollout_training.check('case-00007.wav',run=f.run,output=f.output,job_dir=f.root/'job')
        self.assertEqual(result['case_id'],f.rows[1]['id'])
        self.assertEqual(result['training_updates'],0)
        self.assertEqual(launch.call_args.kwargs['cases'],[f.rows[1]['id']])
        self.assertEqual(launch.call_args.kwargs['steps'],(143,240))
        self.assertEqual(launch.call_args.kwargs['cfgs'],(2.,))
        with self.assertRaisesRegex(ValueError,'exact monitor WAV'):
            rollout_training.check('missing.wav',run=f.run)
        from fireredtts3_local.component_diagnostics import make_plan
        with self.assertRaisesRegex(ValueError,'hybrid adapter'):
            make_plan(f.run,f.output,step=143,cases=[f.rows[0]['id']])


if __name__=='__main__':unittest.main()

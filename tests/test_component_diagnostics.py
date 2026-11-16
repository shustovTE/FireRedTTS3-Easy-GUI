"""Check selective routing against explicit base/delta weights and native rollouts."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from peft.tuners.lora.layer import LoraLayer
from peft.utils.other import ModulesToSaveWrapper
import soundfile as sf
import torch

from fireredtts3_local import cfg_diagnostics as shared
from fireredtts3_local import component_diagnostics as diag
from fireredtts3_local.hybrid import load_adapter
from fireredtts3_local.training_data import file_digest
from fireredtts3_local.training_monitor import generate_case, SETTINGS
import test_cfg_diagnostics as fixtures
from test_hybrid_training import hybrid
from test_training_batch import tiny_core


class ComponentRoutingTests(unittest.TestCase):
    def setUp(self):
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, threads)
        torch.manual_seed(943)
        self.original = tiny_core().eval()
        with torch.no_grad():
            self.original.stop_head.weight.zero_()
            self.original.stop_head.bias.fill_(3.)
        self.model = hybrid(copy.deepcopy(self.original))
        with torch.no_grad():
            for name, value in self.model.named_parameters():
                if value.requires_grad:
                    value.add_(torch.randn_like(value)*.02)
        self.model.eval()
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        reference = Path(temporary.name)/'reference.wav'
        sf.write(reference, np.sin(np.arange(2400)*.1)*.1, 24000)
        self.row = dict(id='test', seed=1235, text='Вопрос?', language='Russian',
                        reference_audio=str(reference), reference_text='Голос.')

    def independent_core(self, active):
        # Independent reference: no selective PEFT routing. Materialize W + BA
        # in a plain core and explicitly copy only the requested dense weights.
        result = copy.deepcopy(self.original)
        with torch.no_grad():
            if 'backbone' in active:
                for name, layer in self.model.get_base_model().named_modules():
                    if isinstance(layer, LoraLayer):
                        delta = layer.lora_B['default'].weight @ layer.lora_A['default'].weight
                        result.get_submodule(name).weight.add_(delta*layer.scaling['default'])
            for name in ('dit', 'dit_head', 'stop_head'):
                if ('stop' if name == 'stop_head' else 'acoustic') in active:
                    getattr(result, name).load_state_dict(
                        getattr(self.model, name).modules_to_save['default'].state_dict())
        return result.eval()

    def test_all_subsets_match_explicit_weights_and_restore_without_mutation(self):
        before = {n: v.clone() for n, v in self.model.state_dict().items()}
        full, _, _ = generate_case(fixtures.TinyPipeline(self.model), self.row)
        outputs = {}
        for name, active in diag.VARIANTS:
            with self.subTest(variant=name):
                audit = diag.select_components(self.model, active)
                expected_core = self.independent_core(active)
                actual, _, metrics = generate_case(fixtures.TinyPipeline(self.model), self.row, trace_flow=True)
                expected, _, expected_metrics = generate_case(fixtures.TinyPipeline(expected_core), self.row, trace_flow=True)
                np.testing.assert_allclose(actual, expected, atol=1e-7, rtol=3e-4)
                np.testing.assert_allclose(metrics['stop_scores'], expected_metrics['stop_scores'], atol=2e-6)
                for module in ('dit_head', 'stop_head'):
                    inputs = torch.linspace(-1, 1, 32)[None]
                    torch.testing.assert_close(getattr(self.model, module)(inputs),
                                               getattr(expected_core, module)(inputs), rtol=0, atol=0)
                self.assertEqual(audit['active_components'], list(active))
                self.assertFalse(any(p.requires_grad for p in self.model.parameters()))
                self.assertFalse(any(m.training for m in self.model.modules()))
                outputs[name] = actual
        self.assertFalse(np.array_equal(outputs['base'], outputs['acoustic_only']))
        self.assertFalse(np.array_equal(outputs['base'], outputs['backbone_only']))
        np.testing.assert_array_equal(outputs['all'], full)
        diag.select_components(self.model, diag.COMPONENTS)
        restored, _, _ = generate_case(fixtures.TinyPipeline(self.model), self.row)
        np.testing.assert_array_equal(restored, full)
        for name, value in self.model.state_dict().items():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)

    def test_reject_invalid_structure_before_switching(self):
        for active in (['unknown'], ['stop', 'stop']):
            with self.assertRaisesRegex(ValueError, 'unique components'):
                diag.select_components(self.model, active)
        with self.assertRaisesRegex(ValueError, 'hybrid adapter'):
            diag.select_components(self.original, ['stop'])
        layer = next(m for m in self.model.modules() if isinstance(m, LoraLayer))
        layer.merge()
        with self.assertRaisesRegex(ValueError, 'unmerged'):
            diag.select_components(self.model, [])
        layer.unmerge()
        self.model.get_base_model().spk_proj_dit = ModulesToSaveWrapper(
            self.model.spk_proj_dit, 'default')
        with self.assertRaisesRegex(ValueError, 'Unexpected dense adapter wrapper'):
            diag.select_components(self.model, [])
        self.assertFalse(layer.disable_adapters)


class ComponentMatrixTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.DiagnosticTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        f = self.fixture
        adapter = f.run/'checkpoint-step-000143'
        core = load_adapter(copy.deepcopy(f.original), adapter).eval()
        previous = f.run/'step143.json'
        results = [dict(id=row['id'], **generate_case(fixtures.TinyPipeline(core), row)[2]) for row in f.rows]
        previous.write_text(json.dumps(dict(step=143, complete=True, settings=SETTINGS,
            fingerprint=f.recipe['generation_benchmark_fingerprint'], results=results)))
        with (f.run/'generation_metrics.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(step=143, report=str(previous)))+'\n')

    def test_full_matrix_controls_resume_and_source_immutability(self):
        f = self.fixture
        plan = diag.make_plan(f.run, f.output)
        self.assertEqual(plan['total_cases'], 16)
        self.assertEqual(plan['cfgs'], [2.])
        self.assertEqual([s['name'] for s in plan['stages']], [n for n, _ in diag.VARIANTS])
        originals = {p: file_digest(p) for p in f.run.rglob('*') if p.is_file()}
        manager = fixtures.FakeManager(f.original)
        report = shared.execute(plan, manager=manager, log=lambda _: None)
        self.assertEqual(manager.loads, 8)
        self.assertTrue(report['complete'])
        self.assertTrue(report['reproduction_controls_complete'])
        self.assertFalse(report['reproduction_review_required'])
        for row in report['results']:
            self.assertEqual(row['active_components'], list(dict(diag.VARIANTS)[row['stage']]))
            self.assertEqual(row['previous_monitor'] is not None, row['stage'] in ('base', 'all'))
            if row['previous_monitor']:
                self.assertEqual(row['previous_monitor']['seconds_delta'], 0)
                self.assertEqual(row['previous_monitor']['rms_drift_delta_db'], 0)
        paused = f.root/'paused'
        paused_plan = diag.make_plan(f.run, paused)
        generated = 0
        def interrupt(*args, **kwargs):
            nonlocal generated
            generated += 1
            if generated == 8:
                raise RuntimeError('simulated interruption')
            return generate_case(*args, **kwargs)
        with patch.object(shared, 'generate_case', side_effect=interrupt):
            with self.assertRaisesRegex(RuntimeError, 'simulated interruption'):
                shared.execute(paused_plan, manager=fixtures.FakeManager(f.original), log=lambda _: None)
        completed = json.loads((paused/'report.json').read_text())['results']
        self.assertEqual(len(completed), 7)
        saved = {paused/p: h for r in completed for p, h in r['artifacts'].items()}
        resumed = shared.execute(paused_plan, resume=True, manager=fixtures.FakeManager(f.original), log=lambda _: None)
        self.assertTrue(resumed['complete'])
        for full, resumed_row in zip(report['results'], resumed['results']):
            self.assertEqual(full['key'], resumed_row['key'])
            np.testing.assert_array_equal(sf.read(f.output/full['float_audio'])[0],
                                          sf.read(paused/resumed_row['float_audio'])[0])
        for path, digest in {**originals, **saved}.items():
            self.assertEqual(file_digest(path), digest)
        self.assertTrue((paused/'diagnostics-report.zip').is_file())
        self.assertTrue((paused/'diagnostics-audio-01.zip').is_file())

    def test_controls_required_and_changed_source_rejected(self):
        f = self.fixture
        with self.assertRaisesRegex(ValueError, 'previous generation report'):
            diag.make_plan(f.run, f.output, step=50)
        plan = diag.make_plan(f.run, f.output)
        source = f.run/'step143.json'
        prior = json.loads(source.read_text())
        prior['results'].pop()
        source.write_text(json.dumps(prior))
        with self.assertRaisesRegex(ValueError, 'missing or duplicate cases'):
            shared.verify_plan(plan)

    def test_cli_passes_component_defaults_to_detached_launcher(self):
        with patch.object(sys, 'argv', ['component_diagnostics', 'start']), \
             patch.object(shared, 'launch', return_value={}) as launch, patch('builtins.print'):
            self.assertEqual(diag.main(), 0)
        self.assertEqual(launch.call_args.kwargs['component_step'], 143)
        self.assertNotIn('step', launch.call_args.kwargs)
        self.assertEqual(launch.call_args.kwargs['job_dir'], diag.DEFAULT_JOB)


if __name__ == '__main__':
    unittest.main()

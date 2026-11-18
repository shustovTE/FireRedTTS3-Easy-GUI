import copy
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from fireredtts3_local import vox_rise_gate as gate, vox_recipe_sweep as sweep


class RiseGateTests(unittest.TestCase):
    def test_percentage_is_on_linear_ratio_and_any_failed_case_vetoes_update(self):
        # The original Base gets quieter. Its ratio is 0.5. The allowed ratio is 0.6.
        base = 20*math.log10(.5)
        rows = [dict(id='a', seed=1236), dict(id='b', seed=1237)]
        baseline = {r['id']:dict(end_minus_start_rms_db=base) for r in rows}
        with patch('fireredtts3_local.vox_recipe_compare.load_saved_base', return_value=({},baseline)), \
                patch.object(gate, 'load_benchmark', return_value=(rows,'digest')):
            config = gate.configuration(20., ['a','b'], [Path('report.json')])
            self.assertAlmostEqual(config['margin_db'],1.5836249209524964)
            with self.assertRaisesRegex(ValueError,'Missing requested'):
                gate.configuration(20., ['absent'], [Path('report.json')])
            for invalid in (float('nan'), -1., float('inf')):
                with self.assertRaises(ValueError):
                    gate.configuration(invalid, ['a'], [Path('report.json')])

        core = torch.nn.Module()
        core.patch_encoder = torch.nn.Linear(1,1).requires_grad_(False)
        core.weight = torch.nn.Parameter(torch.ones(1))
        tts = SimpleNamespace(tts_core=core)
        optimizer = torch.optim.AdamW([core.weight], lr=.1)
        core.weight.grad = torch.ones(1)
        def generate(tts, case, settings):
            self.assertFalse(tts.tts_core.training)
            self.assertFalse(torch.is_grad_enabled())
            self.assertEqual(settings, gate.SETTINGS)
            amplitude = .599 if case['id']=='a' else .601
            # Short clip: disjoint first/last thirds must be used.
            samples = np.repeat([.25, .1, .25*amplitude],100)
            return samples, 100, dict(finish_reason='stop')
        with patch.object(gate, 'generate_case', generate):
            row = gate.try_update(optimizer, [core.weight], tts, config)
        self.assertFalse(row['accepted'])
        self.assertEqual([r['passed'] for r in row['cases']], [True,False])
        self.assertEqual(optimizer.state_dict()['state'], {})
        torch.testing.assert_close(core.weight, torch.ones(1))
        self.assertTrue(core.training)
        self.assertFalse(core.patch_encoder.training)

    def test_rejection_and_evaluation_failure_restore_adamw_including_first_and_later_moments(self):
        parameter = torch.nn.Parameter(torch.tensor([.4, -.2]))
        reference = torch.nn.Parameter(parameter.detach().clone())
        opt = torch.optim.AdamW([parameter], lr=.1)
        control = torch.optim.AdamW([reference], lr=.1)
        for direction, accepted in ((2.,True), (3.,False), (-1.,True)):
            parameter.grad = torch.full_like(parameter, direction)
            with patch.object(gate, 'evaluate', return_value=[{'passed':accepted}]):
                decision = gate.try_update(opt, [parameter], None, {})
            self.assertEqual(decision['accepted'], accepted)
            if accepted:
                reference.grad = torch.full_like(reference, direction)
                control.step()
            torch.testing.assert_close(parameter, reference, atol=0, rtol=0)
            for k, value in control.state[reference].items():
                torch.testing.assert_close(value, opt.state[parameter][k], atol=0, rtol=0)
        before = copy.deepcopy(opt.state_dict())
        parameter.grad = torch.ones_like(parameter)
        with patch.object(gate,'evaluate',side_effect=RuntimeError('synthesis failed')), self.assertRaises(RuntimeError):
            gate.try_update(opt,[parameter],None,{})
        torch.testing.assert_close(parameter, reference, atol=0, rtol=0)
        self.assertEqual(before['param_groups'], opt.state_dict()['param_groups'])
        for k,value in before['state'][0].items():
            torch.testing.assert_close(value, opt.state[parameter][k], atol=0, rtol=0)

    def test_requested_sweep_uses_separate_output_and_pins_gate_in_recipe(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/sweep.vox_recipe_levels.REPORT).write_text('{}')
            p = root/'report.json'
            p.write_text('{}')
            config = dict(percent=20., margin_db=1.5836, cases=[{'id':gate.DEFAULT_CASE}])
            with patch.object(gate,'configuration',return_value=config) as make, \
                    patch.object(sweep,'paired_data',return_value=({},[],{'selection_sha256':'sha'})), \
                    patch.object(sweep.vox_recovery_probe,'select_voice',return_value={}), \
                    patch.object(sweep,'run_slices') as run:
                sweep.fresh_recovery(root,root,[p],four_patch=True,rise_limit_percent=20.)
            make.assert_called_once_with(20., [gate.DEFAULT_CASE], [p])
            _,_,folder,plan,recipe,comparisons = run.call_args.args
            self.assertEqual(folder.name,'lora-base-recovery-4patch-0500-rise20pct')
            self.assertEqual(list(comparisons),[100,500])
            self.assertEqual(recipe['rise_gate'],config)
            self.assertEqual(plan['recipe'],recipe)

"""Native cached rollout vs training forward, with deliberate negative controls."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf
import torch

from fireredtts3_local import training_parity as parity
from fireredtts3_local.training_batch import (Example, FlowSample, batch_conditioning,
                                             flow_inputs, patch_embeddings, predict)
from fireredtts3_local.training_monitor import generate_case, SETTINGS
from test_cfg_diagnostics import TinyPipeline
from test_hybrid_training import hybrid
from test_training_batch import tiny_core


class VariablePipeline(TinyPipeline):
    def __init__(self, core, prompt_patches, text_tokens, maximum):
        super().__init__(core)
        self.prompt_patches, self.text_tokens, self.maximum = prompt_patches, text_tokens, maximum

    def generate(self, *, seed, inference_cfg, stop_threshold, n_timesteps, **_):
        torch.manual_seed(seed)
        prompt = torch.linspace(-.5, .5, self.prompt_patches*8).reshape(1, self.prompt_patches*2, 4)
        output = self.tts_core.generate(spk_emb=torch.ones(1, 6),
            text_tokens=torch.arange(1, self.text_tokens+1)[None], prompt_latents=prompt,
            n_timesteps=n_timesteps, inference_cfg=inference_cfg, stop_threshold=stop_threshold,
            max_gen_steps=self.maximum)
        return output[:, len(prompt[0]):, 0].repeat_interleave(80, dim=1)*.01, 1000


class TrainingParityTests(unittest.TestCase):
    def setUp(self):
        threads = torch.get_num_threads(); torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, threads)
        torch.manual_seed(493)
        self.core = tiny_core().eval()
        with torch.no_grad():
            self.core.stop_head.weight.zero_(); self.core.stop_head.bias.fill_(-3.)
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        reference = Path(temporary.name)/'ref.wav'
        sf.write(reference, np.sin(np.arange(2400)*.1)*.1, 24000)
        self.row = dict(seed=1235, text='Проверка?', language='Russian',
                        reference_audio=str(reference), reference_text='Голос.')

    def adapted(self):
        core = hybrid(copy.deepcopy(self.core))
        with torch.no_grad():
            for p in core.parameters():
                if p.requires_grad:
                    p.add_(torch.randn_like(p)*.01)
        return core.eval()

    def test_cached_native_and_training_agree_for_short_long_prompts_and_hybrid(self):
        for core in (self.core, self.adapted()):
            before = {n: v.clone() for n, v in core.state_dict().items()}
            for prompt, tokens in [(1, 2), (10, 9)]:
                pipeline = VariablePipeline(core, prompt, tokens, 12)
                state = torch.get_rng_state()
                ordinary, rate, normal = generate_case(pipeline, self.row, SETTINGS, trace_flow=True)
                observed, new_rate, metrics = parity.generate_with_parity(pipeline, self.row, SETTINGS, generate_case)
                np.testing.assert_array_equal(ordinary, observed)
                self.assertEqual(rate, new_rate)
                self.assertEqual(normal['stop_scores'], metrics['stop_scores'])
                self.assertEqual(normal['flow_by_patch'], metrics['flow_by_patch'])
                torch.testing.assert_close(torch.get_rng_state(), state, rtol=0, atol=0)
                check = metrics['training_parity']
                self.assertTrue(check['passed'], check)
                self.assertEqual(check['sampled_patch_indices'], [0, 1, 2, 3, 8, 9, 10, 11])
                for value in check['checks'].values():
                    self.assertLess(value['max_abs'], 1e-4)
                for module in (core.dit, core.dit_head, core.stop_head, core.patch_encoder):
                    self.assertFalse(module._forward_hooks)
                    self.assertFalse(module._forward_pre_hooks)
            for name, value in core.state_dict().items():
                torch.testing.assert_close(value, before[name], rtol=0, atol=0)

    def test_deliberately_shifted_training_history_is_detected(self):
        original = parity.flow_inputs
        def shifted(*args, **kwargs):
            current, history, noise, times, drop = original(*args, **kwargs)
            return current, history.roll(1, dims=1), noise, times, drop
        with patch.object(parity, 'flow_inputs', side_effect=shifted):
            _, _, metrics = parity.generate_with_parity(
                VariablePipeline(self.core, 2, 3, 8), self.row, SETTINGS, generate_case)
        self.assertFalse(metrics['training_parity']['passed'])
        self.assertFalse(metrics['training_parity']['checks']['conditional_history_and_noise']['passed'])

    def test_padded_training_batch_matches_separate_native_cached_rollouts(self):
        core = self.adapted()
        traces = []
        constructor = parity.NativeTrace
        def capture(model):
            result = constructor(model)
            traces.append(result)
            return result
        with patch.object(parity, 'NativeTrace', side_effect=capture):
            for prompt, text, maximum in [(1, 2, 6), (10, 11, 12)]:
                parity.generate_with_parity(VariablePipeline(core, prompt, text, maximum), self.row, SETTINGS, generate_case)
        examples, samples, rows = [], [], []
        for trace in traces:
            indexed = {r['index']: r for r in [*trace.first, *trace.last]}
            indices = sorted(indexed)
            selected = [indexed[i] for i in indices]
            rows.extend(selected)
            d = trace.inputs
            examples.append(Example(d['text_tokens'][0], d['prompt_latents'], trace.target, d['spk_emb']))
            samples.append(FlowSample(torch.tensor(indices),
                torch.stack([r['input'][0, -core.patch_size:, :core.redae_dim] for r in selected]),
                torch.zeros(len(selected)), torch.zeros(len(selected), dtype=torch.bool)))
        with torch.inference_mode():
            cond, spk, stops, _, stop_owners, _ = batch_conditioning(
                core, examples, samples, patch_embeddings(core, examples), 'native', False)
            torch.testing.assert_close(cond, torch.cat([r['condition'] for r in rows]), rtol=2e-5, atol=2e-6)
            for i, trace in enumerate(traces):
                torch.testing.assert_close(stops[stop_owners==i][:len(trace.stop_logits)],
                    torch.cat(trace.stop_logits), rtol=2e-5, atol=2e-6)
            for branch in (0, 1):
                for sample in samples:
                    sample.drop.fill_(bool(branch))
                value = predict(core, cond, spk, flow_inputs(core, examples, samples))
                torch.testing.assert_close(value, torch.stack([r['prediction'][branch] for r in rows]), rtol=2e-5, atol=2e-6)

    def test_hooks_and_original_generator_restored_on_generation_error(self):
        pipeline = VariablePipeline(self.core, 1, 2, 12)
        original = self.core.generate
        with patch.object(pipeline, 'generate', side_effect=RuntimeError('synthetic failure')):
            with self.assertRaisesRegex(RuntimeError, 'synthetic failure'):
                parity.generate_with_parity(pipeline, self.row, SETTINGS, generate_case)
        self.assertEqual(self.core.generate, original)
        for module in (self.core.dit, self.core.dit_head, self.core.stop_head):
            self.assertFalse(module._forward_hooks)
            self.assertFalse(module._forward_pre_hooks)

    def test_training_flow_matches_native_single_ode_step_at_nonzero_times(self):
        core = self.adapted()
        count = 3
        current = torch.randn(count, core.patch_size, core.redae_dim)
        history = torch.randn(count, core.history_length, core.redae_dim)
        cond = core.dit_head(torch.randn(count, core.history_patches+1, 32))
        speaker = core.spk_proj_dit(torch.randn(count, 6))
        times = torch.tensor([0., .17, .8])
        native_noise = []
        native_outputs = []
        dt = .125
        # Choose the training interpolation noise to reproduce the exact noisy
        # patch drawn by the native function at each t. This is an algebra test,
        # not a claim that these reconstructed samples follow the training law.
        with torch.inference_mode(), torch.random.fork_rng():
            for i, t in enumerate(times):
                torch.manual_seed(123+i)
                native_noise.append(torch.randn(1, core.patch_size, core.redae_dim))
                torch.manual_seed(123+i)
                native_outputs.append(core._flow_one_step(history[i:i+1], cond[i:i+1], speaker[i:i+1],
                    torch.tensor([t, t+dt]), 2.))
            x = torch.cat(native_noise)
            noise = (x-times[:, None, None]*current)/(1-times[:, None, None])
            conditional = predict(core, cond, speaker, (current, history, noise, times, torch.zeros(count, dtype=torch.bool)))
            unconditional = predict(core, cond, speaker, (current, history, noise, times, torch.ones(count, dtype=torch.bool)))
            expected = x + dt*(3*conditional-2*unconditional)
        torch.testing.assert_close(torch.cat(native_outputs), expected, rtol=2e-5, atol=2e-6)


if __name__ == '__main__':
    unittest.main()

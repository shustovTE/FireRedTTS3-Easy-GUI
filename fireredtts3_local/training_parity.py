"""Observe native generation, then replay identical history through training code.

This checks forward wiring, not the unpublished official training objective or
free-running robustness. No backward, optimizer, RNG replacement or weight edits.
Only the first ODE step (t=0) of the first/last four patches is compared here.
"""
from collections import deque
from unittest.mock import patch

import torch

from .training_batch import (Example, FlowSample, batch_conditioning, flow_inputs,
                             patch_embeddings, predict)
from .training_objective import precision_context


def cpu(value):
    return value.detach().float().cpu().clone()


def difference(actual, expected, *, exact=False):
    actual, expected = cpu(actual), cpu(expected)
    if actual.shape != expected.shape:
        return dict(passed=False, actual_shape=list(actual.shape), expected_shape=list(expected.shape))
    if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
        return dict(passed=False, reason='nonfinite values')
    delta = (actual-expected).reshape(actual.shape[0], -1).double()
    reference = expected.reshape(expected.shape[0], -1).double()
    maximum = delta.abs().amax(dim=1)
    relative = delta.norm(dim=1)/reference.norm(dim=1).clamp_min(1e-12)
    # Full-sequence vs KV-cache BF16 kernels need not be bit-identical. Report
    # every per-patch deviation; these are investigation tolerances, not MOS.
    passed = bool(torch.equal(actual, expected)) if exact else bool(((maximum <= 2e-5) | (relative <= .02)).all())
    return dict(passed=passed, exact=exact, max_abs=float(maximum.max()),
                max_relative_l2=float(relative.max()), per_patch_relative_l2=relative.tolist())


class NativeTrace:
    """Bounded CPU trace: four initial and four final generated patches."""
    def __init__(self, core):
        self.core = core
        self.first = []
        self.last = deque(maxlen=4)
        self.current = None
        self.count = 0
        self.inputs = None
        self.target = None
        self.stop_logits = []
        self.cfg = None

    def condition(self, _module, args, result):
        row = dict(index=self.count, hidden=cpu(args[0]), condition=cpu(result))
        self.count += 1
        self.current = row
        if len(self.first) < 4:
            self.first.append(row)
        self.last.append(row)

    def flow(self, _module, args, kwargs, result):
        row = self.current
        if row is None:
            raise ValueError('DiT called before native conditioning')
        if 'input' in row:
            return
        x = kwargs['x'] if 'x' in kwargs else args[0]
        t = kwargs['t'] if 't' in kwargs else args[1]
        if torch.count_nonzero(t) or x.shape[0] != 2:
            raise ValueError('Parity trace expects native CFG>0 and first flow step t=0')
        row.update(input=cpu(x), prediction=cpu(result[:, -self.core.patch_size:]))

    def generation(self, original, *args, **kwargs):
        if args or self.inputs is not None:
            raise ValueError('Parity expects one unsplit native generation with keyword inputs')
        names = ('spk_emb', 'text_tokens', 'prompt_latents')
        self.inputs = {name: kwargs[name].detach().cpu().clone() for name in names}
        self.cfg = float(kwargs.get('inference_cfg', 2.))
        result = original(**kwargs)
        self.target = cpu(result[:, self.inputs['prompt_latents'].shape[1]:])
        return result

    def check(self):
        core = self.core
        records = {r['index']: r for r in [*self.first, *self.last]}
        indices = sorted(records)
        rows = [records[i] for i in indices]
        if not rows or self.target is None or self.target.shape[1] != self.count*core.patch_size:
            raise ValueError('Native parity trace is incomplete')
        device = next(core.parameters()).device
        data = {k: v.to(device) for k, v in self.inputs.items()}
        example = Example(data['text_tokens'][0], data['prompt_latents'], self.target.to(device), data['spk_emb'])
        native_input = torch.stack([r['input'] for r in rows])
        sample = FlowSample(torch.tensor(indices, device=device),
                            native_input[:, 0, -core.patch_size:, :core.redae_dim].to(device),
                            torch.zeros(len(rows), device=device),
                            torch.zeros(len(rows), dtype=torch.bool, device=device))
        hidden = []
        hook = core.dit_head.register_forward_pre_hook(lambda _m, args: hidden.append(cpu(args[0])))
        try:
            with torch.inference_mode(), precision_context(device, 'native'):
                embeds = patch_embeddings(core, [example])
                cond, spk, stops, _, _, _ = batch_conditioning(
                    core, [example], [sample], embeds, 'native', False)
        finally:
            hook.remove()
        checks = dict(
            backbone_history=difference(hidden[0], torch.cat([r['hidden'] for r in rows])),
            dit_condition=difference(cond, torch.cat([r['condition'] for r in rows])),
            stop_logits=difference(stops[:len(self.stop_logits)], torch.cat(self.stop_logits)))
        predictions = []
        for branch, drop in [('conditional', False), ('unconditional', True)]:
            captured = []
            hook = core.dit.register_forward_pre_hook(lambda _m, args: captured.append(cpu(args[0])))
            try:
                with torch.inference_mode(), precision_context(device, 'native'):
                    sample.drop.fill_(drop)
                    pred = predict(core, cond, spk, flow_inputs(core, [example], [sample]))
            finally:
                hook.remove()
            n = int(drop)
            checks[branch+'_input'] = difference(captured[0], native_input[:, n])
            checks[branch+'_history_and_noise'] = difference(
                captured[0][:, :, :core.redae_dim], native_input[:, n, :, :core.redae_dim], exact=True)
            checks[branch+'_flow'] = difference(pred, torch.stack([r['prediction'][n] for r in rows]))
            predictions.append(cpu(pred))
        cfg = self.cfg
        native_pred = torch.stack([r['prediction'] for r in rows])
        checks['guided_flow'] = difference((1+cfg)*predictions[0]-cfg*predictions[1],
                                           (1+cfg)*native_pred[:, 0]-cfg*native_pred[:, 1])
        return dict(schema='native-training-forward-parity-v1', passed=all(c['passed'] for c in checks.values()),
                    sampled_patch_indices=indices, generated_patches=self.count, flow_time=0.,
                    absolute_tolerance=2e-5, relative_l2_tolerance=.02, checks=checks,
                    note='Eval forward only on identical generated history; t=0, first/last four patches. '
                         'BF16 full-sequence vs cache comparisons allow numerical deviations. '
                         'This does not validate gradients, training data, objective or free-running stability.')


def generate_with_parity(tts, row, settings, generate):
    """Return unchanged native WAV/metrics plus a bounded forward replay audit."""
    core = tts.tts_core
    native = core.get_base_model() if hasattr(core, 'get_base_model') else core
    if core.training or settings.get('do_split') or settings['inference_cfg'] <= 0:
        raise ValueError('Parity requires eval mode, unsplit text and positive CFG')
    trace = NativeTrace(core)
    original = native.generate
    hooks = [core.dit_head.register_forward_hook(trace.condition),
             core.dit.register_forward_hook(trace.flow, with_kwargs=True),
             core.stop_head.register_forward_hook(lambda _m, _a, result: trace.stop_logits.append(cpu(result).flatten()))]
    try:
        with patch.object(native, 'generate', side_effect=lambda *a, **kw: trace.generation(original, *a, **kw)):
            samples, rate, metrics = generate(tts, row, settings, trace_flow=True)
    finally:
        for hook in hooks:
            hook.remove()
    metrics['training_parity'] = trace.check()
    return samples, rate, metrics

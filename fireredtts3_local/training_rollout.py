"""Experimental short, anchored on-policy history augmentation.

Sample detached patches with the current student, then train the next clean
recorded patch conditioned on that history. This is truncated scheduled sampling,
not backpropagation through a whole generated utterance. Local time alignment is
an assumption, bounded by a <=4 patch horizon; it is NOT established by ASR.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math

import torch
import torch.nn.functional as F

from .training_batch import (FlowSample, batch_conditioning, patch_embeddings,
                             predict, set_training)
from .training_objective import precision_context

SCHEMA = 'anchored-own-history-v1'


@dataclass(frozen=True)
class RolloutOptions:
    weight: float = .25
    warmup_steps: int = 20
    grow_every: int = 100
    max_history_patches: int = 4
    min_prefix_patches: int = 6
    terminal_margin_patches: int = 6
    n_timesteps: int = 10
    inference_cfg: float = 2.
    stop_threshold: float = .5

    @classmethod
    def parse(cls, value):
        options = value if isinstance(value, cls) else cls(**value)
        for name in ('warmup_steps', 'grow_every', 'max_history_patches',
                     'min_prefix_patches', 'terminal_margin_patches', 'n_timesteps'):
            x = getattr(options, name)
            if type(x) is not int or x < (0 if name == 'warmup_steps' else 1):
                raise ValueError(f'rollout.{name} must be a valid integer')
        for name in ('weight', 'inference_cfg', 'stop_threshold'):
            x = getattr(options, name)
            if type(x) not in (int, float) or not math.isfinite(x):
                raise ValueError(f'rollout.{name} must be finite')
        if not 0 < options.weight <= 1 or not 0 < options.stop_threshold < 1 or options.inference_cfg < 0:
            raise ValueError('Invalid rollout weight/CFG/stop threshold')
        if options.max_history_patches > 4 or options.min_prefix_patches < 6:
            raise ValueError('Anchored rollout requires <=4 own-history patches and >=6 real prefix patches')
        return options

    def horizon(self, completed_steps):
        if completed_steps < self.warmup_steps:
            return 0
        stage = min(2, (completed_steps-self.warmup_steps)//self.grow_every)
        return min(self.max_history_patches, 2**stage)


def prefix_conditioning(core, examples):
    """Predict after each prefix, with no current/future target in the input.

    Here Example.target holds ONLY observed history. The sampled state is the
    state after its final patch (the same state normally used for terminal STOP).
    No flow_inputs or loss is evaluated against these prefix Examples.
    """
    counts = [e.target.shape[1]//core.patch_size for e in examples]
    device = examples[0].target.device
    samples = [FlowSample(torch.tensor([n], device=device), None, None, None) for n in counts]
    encoded = patch_embeddings(core, examples)
    cond, spk, stops, _, _, _ = batch_conditioning(core, examples, samples, encoded, 'native')
    ends = torch.tensor([n+1 for n in counts], device=device).cumsum(0)-1
    histories = torch.cat([F.pad(torch.cat([e.prompt, e.target], 1),
                                (0, 0, core.history_length, 0))[:, -core.history_length:]
                           for e in examples])
    return cond, spk, histories, stops[ends]


@torch.no_grad()
def sample_native_patches(core, conditions, speaker, history, options):
    """Batched native cosine-time Euler/CFG update; never use a target patch.

    Matches _flow_one_step, including unconditional *acoustic* history and zero
    text/speaker conditioning. Separate function to test against upstream.
    """
    noise = torch.randn(len(history), core.patch_size, history.shape[-1], device=history.device)
    xt = torch.cat([history, noise], 1)
    cond = torch.cat([conditions.repeat_interleave(core.patch_size, 1),
                      speaker[:, None].expand(-1, core.history_length+core.patch_size, -1)], -1)
    times = torch.linspace(0, 1, options.n_timesteps+1, device=history.device)
    times = 1-torch.cos(times*.5*torch.pi)
    for i, time in enumerate(times[:-1]):
        x = torch.cat([xt, cond], -1)
        if options.inference_cfg > 0:
            x = torch.cat([x, torch.cat([xt, torch.zeros_like(cond)], -1)])
        velocity = core.dit(x, time.expand(len(x), 1, 1))[:, -core.patch_size:]
        if options.inference_cfg > 0:
            conditional, unconditional = velocity.chunk(2)
            velocity = (1+options.inference_cfg)*conditional-options.inference_cfg*unconditional
        xt[:, -core.patch_size:] += (times[i+1]-time)*velocity
    result = xt[:, -core.patch_size:]
    if not torch.isfinite(result).all():
        raise FloatingPointError('Nonfinite own-history rollout; no optimizer update was applied')
    return result


@dataclass
class Window:
    owner: int
    anchor: int
    index: int
    prefix: object


@torch.no_grad()
def collect_windows(core, examples, options, horizon, *, anchors=None):
    """Detached, on-policy windows, batched across utterances at every ODE step.

    Recompute only the prefix backbone; no KV cache or whole-rollout graph is
    retained for backward. Re-encode predicted patches instead of using stale
    cached real-patch embeddings. Reject windows hitting native STOP, including
    at the state that would supervise the following real patch.
    """
    windows = []
    for owner, e in enumerate(examples):
        count = e.target.shape[1]//core.patch_size
        last = count-options.terminal_margin_patches-horizon-1
        if last < options.min_prefix_patches:
            continue
        anchor = (int(torch.randint(options.min_prefix_patches, last+1, (), device=e.target.device))
                  if anchors is None else anchors[owner])
        if not options.min_prefix_patches <= anchor <= last:
            raise ValueError('Rollout anchor violates prefix/terminal boundaries')
        prefix = replace(e, target=e.target[:, :anchor*core.patch_size].detach(),
                         target_embeds=(e.target_embeds[:, :anchor].detach()
                                        if e.target_embeds is not None else None), preservation_weight=0.)
        windows.append(Window(owner, anchor, anchor+horizon, prefix))
    attempted, rejected = len(windows), 0
    was_training = core.training
    try:
        core.eval()
        with precision_context(examples[0].target.device, 'native'):
            for offset in range(horizon+1):
                if not windows:
                    break
                prefixes = [w.prefix for w in windows]
                cond, spk, history, stops = prefix_conditioning(core, prefixes)
                if not torch.isfinite(stops).all():
                    raise FloatingPointError('Nonfinite rollout stop scores')
                keep = (stops.sigmoid() < options.stop_threshold).nonzero().flatten()
                rejected += len(windows)-len(keep)
                windows = [windows[i] for i in keep.tolist()]
                if offset == horizon or not windows:
                    break
                generated = sample_native_patches(core, cond[keep], spk[keep], history[keep], options)
                # Pack independent patches along time: PatchEncoder attention is
                # strictly within each patch. Real prefix embeddings stay valid.
                generated_embeds = core.patch_encoder(generated.reshape(1, -1, generated.shape[-1]))
                for i, window in enumerate(windows):
                    e = window.prefix
                    old = e.target_embeds if e.target_embeds is not None else core.patch_encoder(e.target)
                    window.prefix = replace(e,
                        target=torch.cat([e.target, generated[i:i+1]], 1).detach(),
                        target_embeds=torch.cat([old, generated_embeds[:, i:i+1]], 1).detach())
    finally:
        set_training(core, was_training)
    return windows, attempted, rejected


def rollout_loss(tts, examples, options, completed_steps, *, dit_chunk_size=0):
    """Additional flow loss; preservation and STOP are trained on clean data only."""
    options = RolloutOptions.parse(options)
    horizon = options.horizon(completed_steps)
    zero = examples[0].target.new_zeros(())
    metrics = dict(rollout_horizon=float(horizon), rollout_attempted_fraction=0.,
                   rollout_accepted_fraction=0., rollout_stop_rejected_fraction=0.,
                   rollout_flow=0., rollout_contribution=0., rollout_history_rms=0.)
    if not horizon:
        return zero, metrics
    core = tts.tts_core
    windows, attempted, rejected = collect_windows(core, examples, options, horizon)
    metrics.update(rollout_attempted_fraction=attempted/len(examples),
                   rollout_accepted_fraction=len(windows)/len(examples),
                   rollout_stop_rejected_fraction=rejected/len(examples))
    if not windows:
        return zero, metrics
    with precision_context(examples[0].target.device, 'native'):
        cond, spk, history, _ = prefix_conditioning(core, [w.prefix for w in windows])
        # Targets stay real and separate from corrupted history. No generated
        # patch is ever used as the regression target or a preservation teacher.
        current = torch.cat([examples[w.owner].target[:, w.index*core.patch_size:(w.index+1)*core.patch_size]
                             for w in windows]).detach()
        noise = torch.randn_like(current)
        times = torch.rand(len(windows), device=current.device, dtype=current.dtype)
        drop = torch.zeros(len(windows), dtype=torch.bool, device=current.device)
        prediction = predict(core, cond, spk, (current, history, noise, times, drop), dit_chunk_size)
        errors = (prediction-(current-noise).float()).square().mean((1, 2))
    # Rejected/short examples get zero augmentation, not a larger weight on the
    # remaining subset. This reduction survives uneven batches/accumulation.
    contribution = options.weight*errors.sum()/len(examples)
    metrics.update(rollout_flow=errors.detach().sum()/len(examples),
                   rollout_contribution=contribution.detach(),
                   rollout_history_rms=history.detach().square().mean().sqrt())
    return contribution, metrics

"""True variable-length batches with per-example flow/stop/retention losses.

Right padding occurs only AFTER each complete speaker/text/prompt/target sequence.
Padded positions never enter patch selection, acoustic history or loss reduction.
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass

import torch
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

from .training_objective import autocast, precision_context


@dataclass
class Example:
    tokens: torch.Tensor
    prompt: torch.Tensor
    target: torch.Tensor
    speaker: torch.Tensor
    preservation_weight: float = 0.0
    prompt_embeds: torch.Tensor | None = None
    target_embeds: torch.Tensor | None = None


@dataclass
class FlowSample:
    indices: torch.Tensor
    noise: torch.Tensor
    times: torch.Tensor
    drop: torch.Tensor


def set_training(core, training=True):
    core.train(training)
    # Frozen PatchEncoder contains Dropout(0.1). Disabling it is essential for
    # deterministic cache equivalence and teacher/student conditioning.
    if not any(p.requires_grad for p in core.patch_encoder.parameters()):
        core.patch_encoder.eval()


def sample_flow(example, patch_size, patches_per_example, cfg_dropout, generator=None, *, vox_time=False):
    count = example.target.shape[1] // patch_size
    if not count or example.target.shape[1] % patch_size or example.prompt.shape[1] % patch_size or example.prompt.shape[1] < patch_size:
        raise ValueError("Prompt/target must contain complete nonempty patches")
    device = example.target.device
    selected = count if patches_per_example == "all" else min(count, patches_per_example)
    indices = torch.randperm(count, device=device, generator=generator)[:selected]
    shape = (len(indices), patch_size, example.target.shape[-1])
    noise = torch.randn(shape, dtype=example.target.dtype, device=device, generator=generator)
    if vox_time:
        # Vox t runs data -> noise; FireRed's velocity and ODE run noise -> data.
        times = 1 - (torch.randn(len(indices), device=device, generator=generator) - .4).sigmoid()
    else:
        times = torch.rand(len(indices), dtype=example.target.dtype, device=device, generator=generator)
    return FlowSample(indices, noise, times, torch.rand(len(indices), device=device, generator=generator) < cfg_dropout)


def patch_embeddings(core, examples):
    """PatchEncoder attends within each patch; concatenation needs no padding."""
    missing, pieces, sizes = [], [], []
    for i, example in enumerate(examples):
        for field, latents in (("prompt_embeds", example.prompt), ("target_embeds", example.target)):
            if getattr(example, field) is None:
                missing.append((i, field))
                pieces.append(latents)
                sizes.append(latents.shape[1] // core.patch_size)
    result = [{"prompt_embeds": e.prompt_embeds, "target_embeds": e.target_embeds} for e in examples]
    # Cache storage is float32, whereas the encoder's final Linear emits the
    # autocast dtype. Restore it before concatenation: an all-BF16 base must not
    # acquire float32 residual streams solely because features came from disk.
    device_type = examples[0].target.device.type
    if torch.is_autocast_enabled(device_type):
        dtype = torch.get_autocast_dtype(device_type)
        result = [{key: value.to(dtype) if value is not None else None
                   for key, value in item.items()} for item in result]
    if missing:
        packed = core.patch_encoder(torch.cat(pieces, dim=1))
        for (i, field), value in zip(missing, packed.split(sizes, dim=1)):
            result[i][field] = value
    return result


def batch_conditioning(core, examples, samples, embeddings, precision_policy="legacy_bf16",
                       stop_gradient_to_backbone=True):
    """One backbone forward; gather ONLY sampled DiT conditions and valid stops."""
    device = examples[0].target.device
    token_counts = [e.tokens.numel() for e in examples]
    text = core.backbone_llm.embed_tokens(torch.cat([e.tokens.flatten() for e in examples]))
    texts = text.split(token_counts)
    speakers = torch.cat([e.speaker for e in examples])
    spk_llm = core.spk_proj_llm(speakers)
    spk_dit = core.spk_proj_dit(speakers)
    sequences, audio_starts, first_states, target_counts = [], [], [], []
    for i, (example, encoded) in enumerate(zip(examples, embeddings)):
        prompt, target = encoded["prompt_embeds"][0], encoded["target_embeds"][0]
        sequences.append(torch.cat([spk_llm[i:i+1], texts[i], prompt, target], dim=0))
        audio_starts.append(1 + token_counts[i])
        first_states.append(1 + token_counts[i] + len(prompt) - 1)
        target_counts.append(len(target))
    lengths = torch.tensor([len(seq) for seq in sequences], device=device)
    inputs = pad_sequence(sequences, batch_first=True)
    positions = torch.arange(inputs.shape[1], device=device)[None].expand(len(examples), -1)
    mask = positions < lengths[:, None]
    # Native inference autocasts only the backbone; outer context is already
    # configured by the caller for the legacy path.
    with autocast(device) if precision_policy == "native" else contextlib.nullcontext():
        hidden = core.backbone_llm(inputs_embeds=inputs, attention_mask=mask,
                                  position_ids=positions, use_cache=False).last_hidden_state
    patch_owners = torch.cat([torch.full_like(sample.indices, i) for i, sample in enumerate(samples)])
    patch_indices = torch.cat([s.indices for s in samples])
    first = torch.tensor(first_states, device=device)
    audio_start = torch.tensor(audio_starts, device=device)
    offsets = torch.arange(-core.history_patches, 1, device=device)
    gather_positions = first[patch_owners, None] + patch_indices[:, None] + offsets
    valid = gather_positions >= audio_start[patch_owners, None]
    selected = hidden[patch_owners[:, None], gather_positions.clamp_min(0)]
    selected = selected * valid[:, :, None].to(selected.dtype)
    conditions = core.dit_head(selected)
    stop_owners = torch.repeat_interleave(torch.arange(len(examples), device=device),
                                         torch.tensor([n+1 for n in target_counts], device=device))
    stop_indices = torch.cat([torch.arange(n+1, device=device) for n in target_counts])
    stop_hidden = hidden[stop_owners, first[stop_owners] + stop_indices]
    if not stop_gradient_to_backbone:
        stop_hidden = stop_hidden.detach()
    stop_logits = core.stop_head(stop_hidden).float().flatten()
    counts = torch.tensor(target_counts, device=device)
    stop_targets = (stop_indices == counts[stop_owners]).float()
    return conditions, spk_dit[patch_owners], stop_logits, stop_targets, stop_owners, patch_owners


def flow_inputs(core, examples, samples):
    currents, histories, noises, times, masks = [], [], [], [], []
    for example, sample in zip(examples, samples):
        targets = example.target.reshape(-1, core.patch_size, example.target.shape[-1])
        current = targets[sample.indices]
        history = F.pad(torch.cat([example.prompt, example.target], dim=1), (0, 0, core.history_length, 0))
        offsets = example.prompt.shape[1] + sample.indices[:, None] * core.patch_size
        offsets = offsets + torch.arange(core.history_length, device=current.device)[None]
        currents.append(current)
        histories.append(history[0, offsets])
        noises.append(sample.noise)
        times.append(sample.times)
        masks.append(sample.drop)
    return tuple(torch.cat(values) for values in (currents, histories, noises, times, masks))


def predict(core, conditions, speaker, tensors, chunk_size=0):
    if chunk_size:
        # Keep ALL patches, but bound DiT activations. Checkpoint each student
        # chunk so their saved activations do not accumulate until backward.
        from torch.utils.checkpoint import checkpoint
        def chunk(cond, spk, *flow_tensors):
            return predict(core, cond, spk, flow_tensors)
        results = []
        for first in range(0, len(conditions), chunk_size):
            stop = first + chunk_size
            args = (conditions[first:stop], speaker[first:stop], *(t[first:stop] for t in tensors))
            results.append(checkpoint(chunk, *args, use_reentrant=False) if torch.is_grad_enabled()
                           else chunk(*args))
        return torch.cat(results)
    current, history, noise, times, drop = tensors
    noisy = (1-times[:, None, None])*noise + times[:, None, None]*current
    h = conditions.repeat_interleave(core.patch_size, dim=1)
    s = speaker[:, None].expand(-1, core.history_length+core.patch_size, -1)
    condition = torch.cat([h, s], dim=-1) * (~drop)[:, None, None]
    inputs = torch.cat([torch.cat([history, noisy], dim=1), condition], dim=-1)
    return core.dit(inputs, times)[:, -core.patch_size:].float()


def example_mean(values, owners, count):
    sums = values.new_zeros(count).index_add(0, owners, values)
    counts = torch.bincount(owners, minlength=count).to(values.dtype)
    return sums / counts.clamp_min(1)


def reduce_stop_loss(logits, targets, owners, count, mode="mean"):
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    if mode == "mean":
        return example_mean(bce, owners, count)
    if mode != "balanced":
        raise ValueError("Unknown stop loss reduction")
    end = targets.bool()
    return .5 * (example_mean(bce[end], owners[end], count)
                 + example_mean(bce[~end], owners[~end], count))


def stop_diagnostics(logits, targets, owners, count):
    """Teacher-forced endpoint metrics, averaged per utterance, not per frame.

    These expose a missed terminal event hidden by many correct negatives.
    They do not change the objective or prove free-running generation quality.
    Early-stop rates respect the inference minimum of six generated patches.
    """
    probabilities = logits.detach().sigmoid()
    terminal = targets.bool()
    lengths = torch.bincount(owners, minlength=count)
    offsets = lengths.cumsum(0) - lengths
    positions = torch.arange(len(owners), device=owners.device) - offsets[owners]
    eligible = ~terminal & (positions >= 6)
    result = {
        "stop_terminal_probability": probabilities[terminal].mean(),
        "stop_terminal_bce": F.binary_cross_entropy_with_logits(
            logits.detach()[terminal], targets[terminal]),
    }
    for threshold, label in ((.3, "03"), (.5, "05")):
        result[f"stop_terminal_recall_{label}"] = (probabilities[terminal] >= threshold).float().mean()
        early = probabilities.new_zeros(count).index_add(
            0, owners, ((probabilities >= threshold) & eligible).float())
        result[f"stop_early_any_{label}"] = (early > 0).float().mean()
    return result


def batch_flow_loss(tts, examples, patches_per_example=8, cfg_dropout=.1,
                    stop_weight=.05, samples=None, generator=None, dit_chunk_size=0,
                    include_stop_diagnostics=False, precision_policy="legacy_bf16",
                    stop_loss_mode="mean", stop_gradient_to_backbone=True, vox_recipe=False):
    if not examples:
        raise ValueError("Empty training batch")
    core = tts.tts_core
    device = examples[0].target.device
    if samples is None:
        samples = [sample_flow(e, core.patch_size, patches_per_example, cfg_dropout, generator, vox_time=vox_recipe) for e in examples]
    if len(samples) != len(examples):
        raise ValueError("One FlowSample is required per example")
    frozen_patches = not any(p.requires_grad for p in core.patch_encoder.parameters())
    if any(e.prompt_embeds is not None or e.target_embeds is not None for e in examples) and not frozen_patches:
        raise ValueError("Precomputed patch embeddings require a frozen PatchEncoder")
    if frozen_patches:
        core.patch_encoder.eval()
    with precision_context(device, precision_policy):
        embeddings = patch_embeddings(core, examples)
    tensors = flow_inputs(core, examples, samples)
    teacher = None
    retained = [i for i, e in enumerate(examples) if e.preservation_weight > 0]
    if retained:
        if not hasattr(core, "disable_adapter"):
            raise ValueError("Preservation requires frozen base weights and PEFT")
        was_training = core.training
        try:
            core.eval()
            with torch.no_grad(), core.disable_adapter(), precision_context(device, precision_policy):
                subset = [examples[i] for i in retained]
                subset_samples = [samples[i] for i in retained]
                subset_embeds = [{k: v.detach() for k, v in embeddings[i].items()} for i in retained]
                cond, spk, stop, _, stop_owner, patch_owner = batch_conditioning(
                    core, subset, subset_samples, subset_embeds, precision_policy, stop_gradient_to_backbone)
                pred = predict(core, cond, spk, flow_inputs(core, subset, subset_samples), dit_chunk_size)
                teacher = pred, stop, stop_owner, patch_owner
        finally:
            set_training(core, was_training)
    with precision_context(device, precision_policy):
        cond, spk, stop, targets, stop_owner, patch_owner = batch_conditioning(
            core, examples, samples, embeddings, precision_policy, stop_gradient_to_backbone)
        pred = predict(core, cond, spk, tensors, dit_chunk_size)
    count = len(examples)
    if vox_recipe:
        if retained or stop_loss_mode != "mean" or patches_per_example != "all":
            raise ValueError("Vox recipe requires all patches, ordinary STOP and no preservation")
        # Upstream reduces over all valid positions in each microbatch.
        flow = F.mse_loss(pred, (tensors[0]-tensors[2]).float())
        end = F.binary_cross_entropy_with_logits(stop, targets)
        total = flow + stop_weight * end
        metrics = {"flow": flow.detach(), "stop": end.detach(), "loss": total.detach()}
        if include_stop_diagnostics:
            metrics.update(stop_diagnostics(stop, targets, stop_owner, count))
        return total, metrics
    flow = example_mean((pred-(tensors[0]-tensors[2]).float()).square().mean((1, 2)), patch_owner, count)
    stop_loss = reduce_stop_loss(stop, targets, stop_owner, count, stop_loss_mode)
    preservation = flow.new_zeros(count)
    if teacher is not None:
        teacher_pred, teacher_stop, teacher_stop_owner, teacher_patch_owner = teacher
        retained_tensor = torch.tensor(retained, device=device)
        # Order-preserving concatenation matches the teacher's retained subset.
        selected_patches = torch.isin(patch_owner, retained_tensor)
        selected_stops = torch.isin(stop_owner, retained_tensor)
        keep = example_mean((pred[selected_patches]-teacher_pred).square().mean((1, 2)), teacher_patch_owner, len(retained))
        keep = keep + stop_weight * example_mean((stop[selected_stops].sigmoid()-teacher_stop.sigmoid()).square(), teacher_stop_owner, len(retained))
        preservation = preservation.index_copy(0, retained_tensor, keep)
    weights = flow.new_tensor([e.preservation_weight for e in examples])
    total = flow + stop_weight*stop_loss + weights*preservation
    metrics = {"loss": total.detach().mean(), "flow": flow.detach().mean(),
               "stop": stop_loss.detach().mean(), "preservation": preservation.detach().mean()}
    if include_stop_diagnostics:
        metrics.update(stop_diagnostics(stop, targets, stop_owner, count))
    return total.mean(), metrics

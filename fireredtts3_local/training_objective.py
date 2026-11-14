"""Conditional flow training matched to the public Base inference graph.

This is an experimental implementation, not the authors' unpublished trainer.
Joint condition dropout is our explicit choice: it matches the unconditional
branch used by the released CFG inference implementation.
"""
from __future__ import annotations

import contextlib
import hashlib
import os
from pathlib import Path
import uuid

import torch
import torch.nn.functional as F
import torchaudio


def autocast(device):
    return torch.autocast("cuda", dtype=torch.bfloat16) if torch.device(device).type == "cuda" else contextlib.nullcontext()


def adapter_targets(core, profile: str):
    profiles = {
        "backbone": ("backbone_llm.",),
        "backbone_dit": ("backbone_llm.", "dit.", "dit_head"),
    }
    if profile not in profiles:
        raise ValueError(f"Unknown adaptation profile: {profile}")
    names = [name for name, module in core.named_modules()
             if isinstance(module, torch.nn.Linear) and name.startswith(profiles[profile])]
    if not names:
        raise ValueError("No LoRA targets found")
    # Explicit full names protect CAM++, RedAE, PatchEncoder and speaker/stop heads.
    return names


class FeatureCache:
    VERSION = "redae-campp-v1"

    def __init__(self, tts, root=None, model_fingerprint=""):
        self.tts = tts
        self.root = Path(root) if root else None
        if self.root and not model_fingerprint:
            raise ValueError("Disk cache requires a fingerprint of the frozen encoder weights")
        self.namespace = self.VERSION + model_fingerprint
        if self.root:
            self.root.mkdir(parents=True, exist_ok=True)

    def get(self, row):
        key = hashlib.sha256((self.namespace + row["audio_sha256"]).encode()).hexdigest()
        path = self.root / f"{key}.pt" if self.root else None
        if path and path.exists():
            data = torch.load(path, map_location="cpu", weights_only=True)
            if data.get("key") != key:
                raise ValueError(f"Invalid cached feature: {path}")
            return data["latents"].to(self.tts.device), data["speaker"].to(self.tts.device)
        audio, sr = torchaudio.load(row["audio"])
        if audio.shape[0] != 1 or not torch.isfinite(audio).all():
            raise ValueError(f"Expected finite mono audio: {row['audio']}")
        with torch.no_grad():
            audio = torchaudio.functional.resample(audio, sr, self.tts.redae.sample_rate)
            audio = self.tts.redae.pad_to_multiple_of(
                audio, self.tts.redae.downsample_rate * self.tts.tts_core.patch_size
            ).to(self.tts.device)
            latents = self.tts.redae.encode(audio, self.tts.redae.sample_rate).float()
            speaker = self.tts.spk_extractor(audio, self.tts.redae.sample_rate).to(self.tts.device).float()
        # Some upstream encoders use inference_mode internally. Convert their
        # outputs into ordinary tensors before trainable layers save them.
        latents, speaker = latents.clone(), speaker.clone()
        if path:
            temporary = path.with_suffix(f".{uuid.uuid4().hex}.tmp")
            try:
                torch.save({"key": key, "latents": latents.cpu(), "speaker": speaker.cpu()}, temporary)
                os.replace(temporary, path)
            finally:
                temporary.unlink(missing_ok=True)
        return latents, speaker


def conditioning(core, text_tokens, prompt_latents, target_latents, speaker):
    """Return the N next-patch states AND the state after the final patch."""
    prompt_emb = core.patch_encoder(prompt_latents)
    target_emb = core.patch_encoder(target_latents)
    text_emb = core.backbone_llm.embed_tokens(text_tokens)
    speaker_emb = core.spk_proj_llm(speaker).unsqueeze(1)
    embeddings = torch.cat([speaker_emb, text_emb, prompt_emb, target_emb], dim=1)
    hidden = core.backbone_llm(inputs_embeds=embeddings, use_cache=False).last_hidden_state
    start_audio = speaker_emb.shape[1] + text_emb.shape[1]
    first_prediction = start_audio + prompt_emb.shape[1] - 1
    count = target_emb.shape[1]
    # At state j<N a patch must still be generated; only state N means STOP.
    stop_logits = core.stop_head(hidden[:, first_prediction:first_prediction + count + 1]).float().view(-1)
    audio_hidden = hidden[:, start_audio:]
    audio_hidden = F.pad(audio_hidden, (0, 0, core.history_patches, 0))
    windows = audio_hidden.unfold(1, core.history_patches + 1, 1).permute(0, 1, 3, 2)
    first = prompt_emb.shape[1] - 1
    conditions = core.dit_head(windows[0, first:first + count])
    return conditions, core.spk_proj_dit(speaker), stop_logits


def flow_prediction(core, conditions, speaker, histories, noisy, times, indices, drop_mask):
    h = conditions[indices].repeat_interleave(core.patch_size, dim=1)
    s = speaker.unsqueeze(1).expand(len(indices), core.history_length + core.patch_size, -1)
    condition = torch.cat([h, s], dim=-1)
    condition = condition * (~drop_mask).to(condition.dtype)[:, None, None]
    x = torch.cat([histories, noisy], dim=1)
    return core.dit(torch.cat([x, condition], dim=-1), times)[:, -core.patch_size:].float()


def flow_loss(tts, prompt_latents, speaker, target_latents, prompt_text, target_text,
              language, patches_per_example=8, cfg_dropout=0.1, stop_weight=0.05,
              preservation_weight=0.0):
    core = tts.tts_core
    device = target_latents.device
    patch = core.patch_size
    targets = target_latents.reshape(-1, patch, target_latents.shape[-1])
    count = len(targets)
    if count < 1 or prompt_latents.shape[1] < patch:
        raise ValueError("Prompt and target must contain at least one complete patch")
    indices = torch.randperm(count, device=device)[:min(count, patches_per_example)]
    current = targets[indices]
    noise = torch.randn_like(current)
    times = torch.rand(len(indices), device=device, dtype=current.dtype)
    noisy = (1 - times[:, None, None]) * noise + times[:, None, None] * current
    history = F.pad(torch.cat([prompt_latents, target_latents], dim=1), (0, 0, core.history_length, 0))
    offsets = prompt_latents.shape[1] + indices[:, None] * patch + torch.arange(core.history_length, device=device)[None]
    histories = history[0, offsets]
    drop_mask = torch.rand(len(indices), device=device) < cfg_dropout
    tokens = tts._tokenize_text(f"<|{language}|><|sot|>{prompt_text}{target_text}<|eot|>")

    teacher_pred = teacher_stop = None
    if preservation_weight:
        if not hasattr(core, "disable_adapter"):
            raise ValueError("Preservation requires a PEFT adapter on a frozen base")
        was_training = core.training
        try:
            core.eval()
            with torch.no_grad(), core.disable_adapter(), autocast(device):
                cond, spk, teacher_stop = conditioning(core, tokens, prompt_latents, target_latents, speaker)
                teacher_pred = flow_prediction(core, cond, spk, histories, noisy, times, indices, drop_mask)
        finally:
            core.train(was_training)

    with autocast(device):
        cond, spk, stop_logits = conditioning(core, tokens, prompt_latents, target_latents, speaker)
        pred = flow_prediction(core, cond, spk, histories, noisy, times, indices, drop_mask)
    flow = F.mse_loss(pred, current.float() - noise.float())
    stop_targets = torch.zeros_like(stop_logits)
    stop_targets[-1] = 1.0
    stop = F.binary_cross_entropy_with_logits(stop_logits, stop_targets)
    keep = flow.new_zeros(())
    if teacher_pred is not None:
        keep = F.mse_loss(pred, teacher_pred) + stop_weight * F.mse_loss(
            stop_logits.sigmoid(), teacher_stop.sigmoid())
    loss = flow + stop_weight * stop + preservation_weight * keep
    return loss, {"loss": loss.detach(), "flow": flow.detach(), "stop": stop.detach(), "preservation": keep.detach()}

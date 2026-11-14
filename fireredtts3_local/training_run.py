"""Single-GPU adaptation with deterministic validation and resumable sampling."""
from __future__ import annotations

import contextlib
from dataclasses import asdict, dataclass
from functools import wraps
import json
import math
from pathlib import Path
import random

import torch
import torchaudio

from .training_data import (SpeakerPairs, assert_disjoint, dataset_fingerprint,
                            model_identity, load_manifest, split_manifest, write_manifest)
from .training_objective import FeatureCache, adapter_targets, flow_loss


@dataclass
class TrainingOptions:
    profile: str = "backbone"
    validation_manifest: str = ""
    validation_scope: str = "speaker"
    validation_fraction: float = 0.1
    validation_pairs: int = 32
    validate_every: int = 100
    patience: int = 10
    replay_manifest: str = ""
    replay_ratio: float = 0.2
    preservation_weight: float = 0.1
    max_prompt_seconds: float = 12.0
    max_target_seconds: float = 20.0
    patches_per_example: int = 8
    cfg_dropout: float = 0.1
    stop_weight: float = 0.05
    dit_lr_scale: float = 0.25
    cache_dir: str = ".runtime/training-cache"
    attention_backend: str = "sdpa"
    gradient_checkpointing: bool = True

    def validate(self):
        for name, value in asdict(self).items():
            if isinstance(value, (int, float)) and not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.profile not in {"backbone", "backbone_dit"}:
            raise ValueError("profile must be backbone or backbone_dit")
        if self.validation_scope not in {"speaker", "recording"}:
            raise ValueError("validation_scope must be speaker or recording")
        if self.attention_backend not in {"sdpa", "flash_attention_2"}:
            raise ValueError("Unsupported attention backend")
        for name in ("validation_pairs", "validate_every", "patches_per_example"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.patience < 0 or not 0 < self.validation_fraction < 1:
            raise ValueError("Invalid patience/validation_fraction")
        if not 0 <= self.cfg_dropout < 1 or not 0 <= self.replay_ratio < 1:
            raise ValueError("Dropout/replay ratio must be in [0, 1)")
        if not 0 < self.dit_lr_scale <= 1 or self.preservation_weight < 0 or self.stop_weight < 0:
            raise ValueError("Invalid loss weights or DiT learning rate scale")
        if not 0 < self.max_prompt_seconds <= self.max_target_seconds:
            raise ValueError("Require 0 < max_prompt_seconds <= max_target_seconds")
        return self

    @classmethod
    def parse(cls, value=None):
        if isinstance(value, cls):
            return value.validate()
        if isinstance(value, str):
            value = json.loads(value) if value.strip() else {}
        return cls(**(value or {})).validate()


def require_training_hardware():
    if not torch.cuda.is_available():
        raise RuntimeError("Dataset validation passed. Actual FireRedTTS3 training requires an NVIDIA CUDA GPU.")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("This training implementation requires CUDA bfloat16 support")


def prepare_run_data(manifest, options, seed):
    options = TrainingOptions.parse(options)
    rows = load_manifest(manifest)
    if options.validation_manifest:
        train, validation = rows, load_manifest(options.validation_manifest)
    else:
        train, validation = split_manifest(rows, options.validation_fraction, seed, options.validation_scope)
    assert_disjoint(train, validation, options.validation_scope)
    replay = load_manifest(options.replay_manifest) if options.replay_manifest else []
    if replay:
        assert_disjoint(replay, validation, options.validation_scope)
        if not options.replay_ratio:
            raise ValueError("Replay manifest supplied but replay_ratio is zero")
    pair_args = (options.max_prompt_seconds, options.max_target_seconds)
    return (SpeakerPairs(train, *pair_args), SpeakerPairs(validation, *pair_args),
            SpeakerPairs(replay, *pair_args) if replay else None)


@contextlib.contextmanager
def isolated_rng(seed):
    """Validation and synthesis must not alter the subsequent training stream."""
    state = random.getstate()
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    try:
        with torch.random.fork_rng(devices=devices):
            random.seed(seed)
            torch.manual_seed(seed)
            yield
    finally:
        random.setstate(state)


def validate_model(tts, cache, pairs, options, seed):
    was_training = tts.tts_core.training
    values = []
    try:
        tts.tts_core.eval()
        with isolated_rng(seed), torch.no_grad():
            rng = random.Random(seed)
            for _ in range(options.validation_pairs):
                prompt, target = pairs.sample(rng)
                pa, spk = cache.get(prompt)
                ta, _ = cache.get(target)
                _, metrics = flow_loss(tts, pa, spk, ta, prompt["text"], target["text"], target["language"],
                                       options.patches_per_example, 0.0, options.stop_weight, 0.0)
                values.append({k: float(v) for k, v in metrics.items()})
        return {k: sum(v[k] for v in values) / len(values) for k in values[0]}
    finally:
        tts.tts_core.train(was_training)


def exclusive_model(function):
    @wraps(function)
    def run(manager, *args, **kwargs):
        # Keep GUI inference/unload operations from reusing a core mid-update.
        with getattr(manager, "lock", contextlib.nullcontext()):
            return function(manager, *args, **kwargs)
    return run


@exclusive_model
def train_lora(manager, manifest_path, output_dir, rank=16, alpha=32, dropout=0.05,
               learning_rate=1e-5, length_mode="Steps", training_steps=1000,
               training_epochs=10, gradient_accumulation=8, seed=1234, log=print,
               progress_cb=None, save_every=100, resume_checkpoint=None,
               enable_eval_audio=False, eval_text="Проверяем произношение и естественность русской речи.",
               eval_reference_audio=None, eval_reference_transcript="", eval_language="Russian",
               options=None):
    from peft import LoraConfig, PeftModel, get_peft_model
    from torch.utils.tensorboard import SummaryWriter
    from .training import STOP_EVENT
    from .paths import MODEL_DIR

    opts = TrainingOptions.parse(options)
    if min(rank, alpha, training_steps, training_epochs, gradient_accumulation, save_every) < 1:
        raise ValueError("Rank, alpha, lengths, accumulation and save interval must be positive")
    if not 0 <= dropout < 1 or learning_rate <= 0 or not math.isfinite(learning_rate):
        raise ValueError("Invalid dropout/learning rate")
    if length_mode not in {"Steps", "Epochs"}:
        raise ValueError("length_mode must be Steps or Epochs")
    train, validation, replay = prepare_run_data(manifest_path, opts, seed)
    require_training_hardware()
    out = Path(output_dir)
    resume = Path(resume_checkpoint) if resume_checkpoint and str(resume_checkpoint) != "None" else None
    if out.exists() and any(out.iterdir()) and resume is None:
        raise ValueError("Output directory is not empty; choose a new run name or explicitly resume")
    steps_per_epoch = math.ceil(len(train.rows) / gradient_accumulation)
    total = training_steps if length_mode == "Steps" else steps_per_epoch * training_epochs
    recipe = {
        "schema": 2, "options": asdict(opts), "rank": rank, "alpha": alpha, "dropout": dropout,
        "learning_rate": learning_rate, "accumulation": gradient_accumulation,
        "seed": seed, "length_mode": length_mode, "total_steps": total,
        "train_fingerprint": dataset_fingerprint(train.rows),
        "validation_fingerprint": dataset_fingerprint(validation.rows),
        "replay_fingerprint": dataset_fingerprint(replay.rows) if replay else None,
    }
    state = None
    if resume:
        state = torch.load(resume / "trainer_state.pt", map_location="cpu", weights_only=True)
        if state.get("recipe") != recipe:
            raise ValueError("Resume recipe/data differ. Resume with the original recipe; legacy all-linear checkpoints are not resumable here.")
    out.mkdir(parents=True, exist_ok=True)
    write_manifest(out / "train.jsonl", train.rows)
    write_manifest(out / "validation.jsonl", validation.rows)
    if replay:
        write_manifest(out / "replay.jsonl", replay.rows)
    (out / "recipe.json").write_text(json.dumps(recipe, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"[training] {len(train.groups)} speaker/language groups; {len(validation.rows)} validation clips ({opts.validation_scope}); profile={opts.profile}")
    log("[training] Epoch means a speaker-balanced sampling budget, not an exhaustive pass over clips.")
    STOP_EVENT.clear()
    random.seed(seed)
    torch.manual_seed(seed)
    rng = random.Random(seed)
    writer = None
    try:
        # Always start from the clean base, never an inference adapter left in memory.
        manager.unload(log)
        tts = manager.load_base(log=log, attention_backend=opts.attention_backend, torch_compile=False)
        base_fingerprint = model_identity(MODEL_DIR)
        if state and state.get("base_fingerprint") != base_fingerprint:
            raise ValueError("The base weights/tokenizer have changed since this checkpoint")
        for module in (tts.redae, tts.spk_extractor):
            module.eval()
            module.requires_grad_(False)
        core = tts.tts_core
        targets = adapter_targets(core, opts.profile)
        if resume:
            core = PeftModel.from_pretrained(core, str(resume), is_trainable=True)
        else:
            core = get_peft_model(core, LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=dropout,
                                                 target_modules=targets, bias="none"))
        tts.tts_core = core
        if opts.gradient_checkpointing:
            core.backbone_llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        named = [(n, p) for n, p in core.named_parameters() if p.requires_grad]
        if not named or any("lora_" not in n for n, _ in named):
            raise RuntimeError("Expected only LoRA parameters to be trainable")
        (out / "trainable_parameters.json").write_text(json.dumps([n for n, _ in named], indent=2))
        acoustic = [p for n, p in named if ".dit." in n or ".dit_head." in n]
        backbone = [p for n, p in named if ".dit." not in n and ".dit_head." not in n]
        groups = [{"params": backbone, "lr": learning_rate}]
        if acoustic:
            groups.append({"params": acoustic, "lr": learning_rate * opts.dit_lr_scale})
        optimizer = torch.optim.AdamW(groups, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01)
        warmup = max(1, min(100, int(total * 0.05)))
        def schedule(step):
            if step < warmup:
                return (step + 1) / warmup
            fraction = min(1.0, (step - warmup) / max(1, total - warmup))
            return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * fraction))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
        cache = FeatureCache(tts, opts.cache_dir or None, base_fingerprint)
        step, best, bad = 0, float("inf"), 0
        if state:
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            rng.setstate(state["sampler_rng"])
            random.setstate(state["python_rng"])
            torch.set_rng_state(state["torch_rng"])
            torch.cuda.set_rng_state_all(state["cuda_rng"])
            step, best, bad = state["step"], state["best_validation"], state["bad_validations"]
        writer = SummaryWriter(str(out / "tensorboard"))
        metadata = {"experimental": True, "recipe": recipe, "base_model": "FireRedTeam/FireRedTTS3", "base_fingerprint": base_fingerprint,
                    "selection_metric": "held-out flow + stop loss; NOT a guarantee of voice similarity"}

        def save(path):
            path.mkdir(parents=True, exist_ok=True)
            core.save_pretrained(str(path))
            torch.save({"recipe": recipe, "base_fingerprint": base_fingerprint, "step": step, "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(), "sampler_rng": rng.getstate(),
                        "python_rng": random.getstate(), "torch_rng": torch.get_rng_state(),
                        "cuda_rng": torch.cuda.get_rng_state_all(), "best_validation": best,
                        "bad_validations": bad}, path / "trainer_state.pt")
            (path / "firered_lora_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        def evaluate():
            nonlocal best, bad
            metrics = validate_model(tts, cache, validation, opts, seed + 10000)
            if not all(math.isfinite(x) for x in metrics.values()):
                raise FloatingPointError("Non-finite validation metrics")
            for key, value in metrics.items():
                writer.add_scalar("validation/" + key, value, step)
            with (out / "validation_metrics.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"step": step, **metrics}) + "\n")
            log(f"[validation] step={step} loss={metrics['loss']:.6f} flow={metrics['flow']:.6f} stop={metrics['stop']:.6f}")
            if metrics["loss"] < best:
                best, bad = metrics["loss"], 0
                save(out / "best-validation")
            else:
                bad += 1
            return metrics

        def preview():
            if not enable_eval_audio:
                return
            if not eval_reference_audio or not eval_reference_transcript.strip():
                raise ValueError("Preview requires a reference recording and its exact transcript")
            was_training = core.training
            try:
                core.eval()
                with isolated_rng(seed), torch.inference_mode():
                    ref, sr = torchaudio.load(eval_reference_audio)
                    wav, rate = tts.generate(language=eval_language, prompt_text=eval_reference_transcript,
                                            prompt_audio=ref, prompt_audio_sr=sr, text=eval_text,
                                            seed=seed, do_split=False)
                folder = out / "eval_audio"
                folder.mkdir(exist_ok=True)
                torchaudio.save(str(folder / f"step-{step:06d}.wav"), wav.cpu(), rate)
            finally:
                core.train(was_training)

        core.train()
        if not state:
            evaluate()
            preview()
        while step < total and not STOP_EVENT.is_set():
            optimizer.zero_grad(set_to_none=True)
            metrics_sum = {}
            used = 0
            # Epoch mode's final update divides by its ACTUAL micro-batch count.
            epoch_position = step % steps_per_epoch
            batch_size = (min(gradient_accumulation, len(train.rows) - epoch_position * gradient_accumulation)
                          if length_mode == "Epochs" else gradient_accumulation)
            for _ in range(batch_size):
                if STOP_EVENT.is_set():
                    break
                is_replay = replay is not None and rng.random() < opts.replay_ratio
                prompt, target = (replay if is_replay else train).sample(rng)
                pa, speaker = cache.get(prompt)
                ta, _ = cache.get(target)
                # When replay exists, preserve on replay only; otherwise regularize
                # the main corpus against the original model with the explicit weight.
                keep = opts.preservation_weight if is_replay or replay is None else 0.0
                loss, metrics = flow_loss(tts, pa, speaker, ta, prompt["text"], target["text"], target["language"],
                                          opts.patches_per_example, opts.cfg_dropout, opts.stop_weight, keep)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite training loss at step {step}, utterance {target['utt_id']}")
                loss.backward()
                used += 1
                for key, value in metrics.items():
                    metrics_sum[key] = metrics_sum.get(key, 0.0) + float(value)
            if not used:
                break
            for _, parameter in named:
                if parameter.grad is not None:
                    parameter.grad.div_(used)
            norm = torch.nn.utils.clip_grad_norm_([p for _, p in named], 0.5, error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            step += 1
            for key, value in metrics_sum.items():
                writer.add_scalar("train/" + key, value / used, step)
            writer.add_scalar("train/grad_norm", float(norm), step)
            writer.add_scalar("train/learning_rate", optimizer.param_groups[0]["lr"], step)
            message = f"[training] step={step}/{total} loss={metrics_sum['loss']/used:.6f}"
            log(message)
            if progress_cb:
                progress_cb(step, total, message)
            due_save = (step % save_every == 0 if length_mode == "Steps"
                        else step % (save_every * steps_per_epoch) == 0)
            if step % opts.validate_every == 0 or step == total or due_save:
                evaluate()
                if due_save or step == total:
                    save(out / f"checkpoint-step-{step:06d}")
                    preview()
                if opts.patience and bad >= opts.patience:
                    log("[training] Early stopping: validation loss has not improved")
                    break
        save(out)
        log("[training] Compare best-validation and baseline on unseen voices before deployment.")
        return out, step if length_mode == "Steps" else step // steps_per_epoch
    finally:
        if writer:
            writer.close()
        manager.unload(log)

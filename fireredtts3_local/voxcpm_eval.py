"""Whole-utterance, cloned-reference evaluation with the pinned native VoxCPM2.

Run Base and an official LoRA checkpoint into different output directories using
the same benchmark. This records reproducible audio, not a perceptual pass gate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import soundfile as sf

from .native_comparison import signal_levels
from .workspace_paths import local_path
from .voxcpm_training import (DEFAULT_MODEL, DEFAULT_SOURCE, MODEL_ID, MODEL_MARKER,
                              MODEL_REVISION, SOURCE_COMMIT, digest, verify_model,
                              verify_source, write_json)


def _text_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_benchmark(path, seed=None):
    """Keep exact strings; resolve references relative to the benchmark file."""
    path = local_path(path).resolve(strict=True)
    rows, ids = [], set()
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or any(not isinstance(row.get(k), str) or not row[k].strip()
                                           for k in ("id", "text", "reference_text", "reference_audio")):
            raise ValueError(f"Benchmark line {number} needs id, text and exact reference audio/text")
        if row["id"] in ids:
            raise ValueError(f"Duplicate benchmark id: {row['id']}")
        ids.add(row["id"])
        chosen_seed = row.get("seed", 1234) if seed is None else seed
        if type(chosen_seed) is not int or not 0 <= chosen_seed < 2**32:
            raise ValueError("Generation seed must be an integer in 0..2**32-1")
        reference = local_path(row["reference_audio"])
        reference = (reference if reference.is_absolute() else path.parent / reference).resolve(strict=True)
        audio, sr = sf.read(reference, dtype="float32", always_2d=True)
        if audio.shape[1] != 1 or not len(audio) or sr <= 0 or not np.isfinite(audio).all():
            raise ValueError(f"Reference must be finite nonempty mono audio: {reference}")
        rows.append({**row, "seed": chosen_seed, "reference_audio": str(reference),
                     "text_sha256": _text_hash(row["text"]),
                     "reference_text_sha256": _text_hash(row["reference_text"]),
                     "reference_sha256": digest(reference), "reference_samples": len(audio),
                     "reference_sample_rate": sr, "reference_seconds": len(audio) / sr})
    if not rows:
        raise ValueError("Benchmark is empty")
    return rows


def adapter_info(adapter, model):
    if adapter is None:
        return None
    path = local_path(adapter).resolve(strict=True)
    config_path = path / "lora_config.json"
    info = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(info.get("lora_config"), dict) or not info["lora_config"]:
        raise ValueError("Expected an official checkpoint directory with lora_config.json")
    declared = info.get("base_model")
    if not isinstance(declared, str) or (declared != MODEL_ID and local_path(declared).resolve() != model):
        raise ValueError("Adapter declares a different base model; use the checkpoint trained on this pinned base")
    weights = next((path / name for name in ("lora_weights.safetensors", "lora_weights.ckpt")
                    if (path / name).is_file()), None)
    if weights is None:
        raise ValueError("Official LoRA checkpoint has no lora_weights.safetensors or lora_weights.ckpt")
    return {"path": str(path), "weights": str(weights), "config": info["lora_config"],
            "declared_base_model": declared,
            "files": {config_path.name: digest(config_path), weights.name: digest(weights)}}


def _attach_adapter(tts, adapter):
    if adapter is None:
        return
    loaded, skipped = tts.load_lora_weights(adapter["weights"])
    expected = {name for name, _ in tts.named_parameters() if "lora_" in name}
    if not expected or skipped or set(loaded) != expected:
        raise ValueError("LoRA loading was incomplete or contained unexpected tensors; refusing a partial adapter")
    for name, expected_hash in adapter["files"].items():
        if digest(Path(adapter["path"]) / name) != expected_hash:
            raise ValueError("LoRA checkpoint changed while loading")


def _load_model(model, source, adapter):
    import torch
    import voxcpm.model.voxcpm2 as native
    if not Path(native.__file__).resolve().is_relative_to(source / "src"):
        raise ValueError("Imported VoxCPM does not come from the verified source checkout; use .venv-voxcpm2")
    if not torch.cuda.is_available():
        raise RuntimeError("Native VoxCPM2 evaluation requires the CUDA host; no generation was run")
    config = native.LoRAConfig(**adapter["config"]) if adapter else None
    tts = native.VoxCPM2Model.from_local(str(model), optimize=False, training=False,
                                      device="cuda", lora_config=config)
    _attach_adapter(tts, adapter)
    return tts.eval()


def _context_audit(tts, row, max_steps):
    tokens = len(tts.text_tokenizer(row["reference_text"] + row["text"]))
    target_tokens = len(tts.text_tokenizer(row["text"]))
    if target_tokens < 1:
        raise ValueError("Tokenizer produced no target tokens")
    # Combined mode: [ref_start, reference patches, ref_end], full text,
    # audio_start, prompt patches. Both audio inputs are the same recording.
    resampled = math.ceil(row["reference_samples"] * tts._encode_sample_rate / row["reference_sample_rate"])
    reference_patches = math.ceil(resampled / (tts.patch_size * tts.chunk_size))
    prefix = tokens + reference_patches * 2 + 3
    limit = tts.config.max_length
    if prefix + max_steps > limit:
        raise ValueError(f"Full text/reference plus requested generation needs {prefix + max_steps} tokens, "
                         f"but context is {limit}; nothing was truncated")
    return {"text_and_prompt_tokens": tokens, "target_tokens": target_tokens,
            "reference_patches": reference_patches, "prefix_tokens": prefix, "context_limit": limit}


def generate_case(tts, row, settings):
    import torch
    audit = _context_audit(tts, row, settings["max_steps"])
    if digest(row["reference_audio"]) != row["reference_sha256"]:
        raise ValueError("Benchmark reference changed after validation")
    stops = []
    def observe(_module, _inputs, logits):
        if logits.shape[-1] != 2 or logits.numel() != 2:
            raise ValueError("Unexpected native two-class STOP output")
        stops.append(int(logits.detach().float().argmax(dim=-1).reshape(-1)[0]))
    hook = tts.stop_head.register_forward_hook(observe)
    started = time.monotonic()
    try:
        with torch.inference_mode():
            waveform = tts.generate(
                target_text=row["text"], prompt_text=row["reference_text"],
                prompt_wav_path=row["reference_audio"], reference_wav_path=row["reference_audio"],
                min_len=settings["min_steps"], max_len=settings["max_steps"],
                inference_timesteps=settings["inference_timesteps"], cfg_value=settings["cfg_value"],
                retry_badcase=False, retry_badcase_max_times=1,
                # Upstream applies its text-ratio length cap even with retry=False.
                # This value makes the explicit max_steps the only length cap.
                retry_badcase_ratio_threshold=float(settings["max_steps"]),
                trim_silence_vad=False, seed=row["seed"])
    finally:
        hook.remove()
    if not torch.is_tensor(waveform) or waveform.ndim != 2 or waveform.shape[0] != 1:
        raise ValueError("Native model returned an unexpected mono waveform shape")
    samples = waveform[0].detach().float().cpu().numpy()
    levels = signal_levels(samples, tts.sample_rate)
    if not stops or len(stops) > settings["max_steps"]:
        raise ValueError("Native STOP trace is missing or inconsistent")
    if getattr(tts, "last_successful_seed", None) != row["seed"]:
        raise ValueError("Native generation did not retain the requested seed")
    stopped = len(stops) - 1 > settings["min_steps"] and stops[-1] == 1
    if not stopped and len(stops) != settings["max_steps"]:
        raise ValueError("Generation ended without a native STOP or the explicit generation limit")
    n = min(10 * tts.sample_rate, len(samples) // 3)
    if n:
        rms = lambda a: 20 * math.log10(max(float(np.sqrt(np.mean(a.astype(np.float64)**2))), 1e-12))
        levels["end_minus_start_rms_db"] = rms(samples[-n:]) - rms(samples[:n])
        levels["rms_window_seconds"] = n / tts.sample_rate
    return samples, {**levels, "context": audit, "elapsed_seconds": time.monotonic() - started,
                     "generated_patches": len(stops), "stop_classes": stops,
                     "finish_reason": "stop" if stopped else "max_steps",
                     "failures": [] if stopped else ["generation_limit"]}


def full_checkpoint_info(directory, base, base_files):
    """Native from_local loads with strict=False. Reject partial weights before inference."""
    from safetensors import safe_open
    directory = local_path(directory).resolve(strict=True)
    step = json.loads((directory / "training_state.json").read_text())["step"]
    if type(step) is not int or step < 0:
        raise ValueError("Expected an official full training checkpoint with a numbered step")
    if any((directory / name).exists() for name in ("lora_config.json", "audiovae.safetensors")):
        raise ValueError("Expected a full checkpoint with the original AudioVAE")
    files = {name: digest(directory / name) for name in base_files}
    for name in base_files.keys() - {"model.safetensors"}:
        if files[name] != base_files[name]:
            raise ValueError(f"Full checkpoint differs from pinned base asset: {name}")
    with safe_open(base / "model.safetensors", framework="pt") as original, safe_open(
        directory / "model.safetensors", framework="pt"
    ) as trained:
        if set(trained.keys()) != set(original.keys()) or any(
            trained.get_slice(k).get_shape() != original.get_slice(k).get_shape() for k in original.keys()
        ):
            raise ValueError("Full checkpoint weights do not cover the base architecture")
    return directory, step, files


def evaluate(benchmark, model, output, adapter=None, *, source=DEFAULT_SOURCE, full_checkpoint=None,
             seed=None, max_steps=1000, cfg=2., steps=10):
    if type(max_steps) is not int or max_steps < 4 or type(steps) is not int or steps < 1:
        raise ValueError("max_steps must be >=4 and steps must be >=1")
    if isinstance(cfg, bool) or not isinstance(cfg, (int, float)) or not math.isfinite(cfg) or cfg < 0:
        raise ValueError("CFG must be finite and nonnegative")
    benchmark, model, output = (local_path(p).resolve() for p in (benchmark, model, output))
    if output.exists():
        raise FileExistsError(f"Evaluation output exists; use a new directory: {output}")
    rows = load_benchmark(benchmark, seed)
    source = verify_source(local_path(source))
    verify_model(model)
    receipt = json.loads((model / MODEL_MARKER).read_text(encoding="utf-8"))
    if full_checkpoint is not None and adapter is not None:
        raise ValueError("Select one full checkpoint or one LoRA adapter")
    adapter = adapter_info(adapter, model)
    base = model
    step, files = None, receipt["files"]
    if full_checkpoint is not None:
        model, step, files = full_checkpoint_info(full_checkpoint, base, files)
    settings = {"clone_mode": "reference_plus_prompt", "split_text": False, "normalize_text": False,
                "denoise": False, "trim_silence_vad": False, "retry_badcase": False, "max_attempts": 1,
                "min_steps": 2, "max_steps": max_steps, "cfg_value": float(cfg), "inference_timesteps": steps,
                "retry_badcase_ratio_threshold": float(max_steps), "optimize": False, "device": "cuda"}
    report = {"schema": "voxcpm2-native-eval-v1", "complete": False, "results": [],
              "benchmark": str(benchmark), "benchmark_sha256": digest(benchmark),
              "source": str(source), "source_commit": SOURCE_COMMIT, "model": str(model),
              "model_id": MODEL_ID if step is None else None,
              "model_revision": MODEL_REVISION if step is None else None, "model_files": files,
              "adapter": adapter, "settings": settings,
              "quality_assessment": "Listening and content/voice checks are required; these metrics are not a perceptual gate."}
    if step is not None:
        report["full_checkpoint"] = {"step": step, "path": str(model), "base_path": str(base),
                                     "base_id": MODEL_ID, "base_revision": MODEL_REVISION}
    tts = _load_model(model, source, adapter)
    settings["patch_seconds"] = tts.patch_size * tts._decode_chunk_size / tts.sample_rate
    settings["max_generated_seconds"] = max_steps * settings["patch_seconds"]
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "report.json", report)
    for index, row in enumerate(rows):
        result = {**row, "index": index, "characters": len(row["text"])}
        try:
            samples, metrics = generate_case(tts, row, settings)
            name = f"case-{index:05d}"
            raw, listening = output / f"{name}.float.wav", output / f"{name}.wav"
            gain = min(1., .98 / max(metrics["peak"], 1e-12))
            sf.write(raw, samples, tts.sample_rate, subtype="FLOAT")
            sf.write(listening, samples * gain, tts.sample_rate, subtype="PCM_16")
            result.update(metrics, status="generated", float_audio=raw.name, audio=listening.name,
                          float_audio_sha256=digest(raw), audio_sha256=digest(listening), pcm_gain=gain)
            print(f"[voxcpm-eval] {row['id']}: {metrics['seconds']:.2f}s; {metrics['finish_reason']}; "
                  f"raw peak={metrics['peak']:.3f}", flush=True)
        except Exception as exc:
            result.update(status="error", failures=["generation_error"],
                          error_type=type(exc).__name__, error=str(exc))
            print(f"[voxcpm-eval] {row['id']}: ERROR {type(exc).__name__}: {exc}", flush=True)
        report["results"].append(result)
        report["completed_cases"] = len(report["results"])
        write_json(output / "report.json", report)
    report["complete"] = True
    report["all_generated"] = all(r["status"] == "generated" for r in report["results"])
    report["execution_failures"] = {r["id"]: r["failures"] for r in report["results"] if r["failures"]}
    write_json(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--source", default=str(DEFAULT_SOURCE))
    weights = parser.add_mutually_exclusive_group()
    weights.add_argument("--full-checkpoint", help="Numbered checkpoint directory from official full fine-tuning")
    weights.add_argument("--adapter", help="Official checkpoint directory; omit for the original Base")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, help="Override every case seed; otherwise keep benchmark seeds")
    parser.add_argument("--max-steps", type=int, default=1000, help="Audio patches, 160 seconds at the pinned model's 6.25 Hz")
    parser.add_argument("--cfg", type=float, default=2.)
    parser.add_argument("--steps", type=int, default=10, help="Flow solver steps per audio patch")
    args = parser.parse_args()
    result = evaluate(**vars(args))
    print(json.dumps({"report": str(local_path(args.output).resolve() / "report.json"),
                      "complete": result["complete"], "execution_failures": result["execution_failures"]}, indent=2))
    if result["execution_failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

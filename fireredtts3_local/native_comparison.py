"""Compare the upstream native generator and SGLang using one merged release."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import time
import zipfile

CASES = ("cities", "timofey")


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def prepare_native_view(base, release, view):
    """Use identical merged core shards with the original native core config.

    The Hub's root config describes the SGLang pipeline, not the native core.
    No tensors are converted or merged again, and no big files are copied.
    """
    base, release, view = Path(base).resolve(), Path(release).resolve(), Path(view)
    core = view / "fireredtts3_base"
    core.mkdir(parents=True, exist_ok=False)
    (core / "config.json").write_bytes((base / "fireredtts3_base/config.json").read_bytes())
    shards = sorted(release.glob("model*.safetensors"))
    if not shards:
        raise ValueError("Merged release has no core safetensors")
    index = release / "model.safetensors.index.json"
    for path in shards + ([index] if index.is_file() else []):
        (core / path.name).symlink_to(path)
    for name in ("redae", "campp", "text_tokenizer"):
        (view / name).symlink_to(base / name, target_is_directory=True)
    return view


def signal_levels(audio, sample_rate):
    """Unweighted signal RMS/peak, not perceptual loudness or a quality score."""
    import numpy as np
    audio = np.asarray(audio, dtype=np.float64)
    if audio.ndim != 1 or not audio.size or not np.isfinite(audio).all() or sample_rate < 1:
        raise ValueError("Expected nonempty finite mono audio")
    windows = []
    for start in range(0, len(audio), sample_rate):
        part = audio[start:start + sample_rate]
        rms = float(np.sqrt(np.mean(part * part)))
        windows.append(dict(start_seconds=start / sample_rate, seconds=len(part) / sample_rate,
                            rms_dbfs=float(20 * np.log10(max(rms, 1e-12))),
                            peak=float(np.max(np.abs(part)))))
    return dict(seconds=len(audio) / sample_rate, sample_rate=sample_rate,
                peak=float(np.max(np.abs(audio))),
                fraction_at_or_above_full_scale=float(np.mean(np.abs(audio) >= 1.)),
                fraction_near_full_scale=float(np.mean(np.abs(audio) >= 32767 / 32768)),
                windows=windows,
                note="One-second unweighted RMS; pauses and phonemes affect levels. No audio normalization.")


def run_native(root, release, base, seed=1234, threshold=.5):
    import os
    import soundfile as sf
    import torch
    from .release import verify_release
    from .training_data import file_digest, model_identity

    root, release, base = Path(root).resolve(), Path(release).resolve(), Path(base).resolve()
    if not 0 < threshold < 1:
        raise ValueError("Stop threshold must be between 0 and 1")
    print("[native-compare] Verifying merged release and original base hashes...", flush=True)
    manifest = verify_release(release)
    deployed = json.loads((root / "deployed_manifest.json").read_text())
    if deployed != manifest:
        raise ValueError("SGLang and local export name different releases; compare the same weights")
    provenance = json.loads((release / "export_provenance.json").read_text())
    if (file_digest(root / "deployed_provenance.json") != manifest["files"]["export_provenance.json"]["sha256"]
            or model_identity(base) != provenance["base_fingerprint"]):
        raise ValueError("Deployed provenance or native base assets differ from the merged export")
    reference = json.loads((root / "reference.json").read_text())
    if (file_digest(root / "reference.wav") != reference["reference_sha256"]
            or not reference["reference_text"].strip()):
        raise ValueError("Invalid saved reference WAV/transcript")
    texts = {name: (root / "texts" / f"{name}.txt").read_text(encoding="utf-8").strip() for name in CASES}
    if any(not text or len(text) > 2000 for text in texts.values()):
        raise ValueError("Each complete test text must contain 1..2000 characters")
    if not torch.cuda.is_available():
        raise RuntimeError("Native generation requires the host CUDA GPU")
    audio, rate = sf.read(root / "reference.wav", dtype="float32", always_2d=True)
    if audio.shape[1] != 1:
        raise ValueError("Expected the stored mono reference; do not change channels between backends")
    prompt = torch.from_numpy(audio.T.copy())
    output = root / "native"
    output.mkdir(exist_ok=False)
    settings = dict(release_id=manifest["release_id"], reference_sha256=reference["reference_sha256"],
                    seed=seed, language="Russian", stop_threshold=threshold, n_timesteps=10,
                    inference_cfg=2.0, min_patches=6, max_patches=400, split_text=False,
                    text_normalization=False, attention="sdpa", torch_compile=False,
                    weights="exact merged core shards; original verified native tokenizer/RedAE/CAM++",
                    note="Same seed and weights do not imply identical RNG streams or numerical execution across backends.")
    write_json(root / "comparison_settings.json", settings)
    os.environ["FIRERED_ATTN_IMPLEMENTATION"] = "sdpa"
    from .model_manager import ModelManager
    ModelManager._ensure_upstream_namespace()
    from fireredtts3.llm.fireredtts3_base import FireRedTTS3Base
    report = dict(complete=False, results=[])
    with tempfile.TemporaryDirectory(prefix="firered-native-view-") as temporary:
        view = prepare_native_view(base, release, Path(temporary) / "model")
        print("[native-compare] Loading merged weights through the upstream native generator...", flush=True)
        tts = FireRedTTS3Base(str(view))
        try:
            for module in (tts.tts_core, tts.redae, tts.spk_extractor):
                module.eval()
            for name, text in texts.items():
                scores = []
                handle = tts.tts_core.stop_head.register_forward_hook(
                    lambda _module, _args, value: scores.append(float(torch.sigmoid(value.detach().float()).reshape(-1)[-1])))
                started = time.monotonic()
                print(f"[native-compare] {name}: {len(text)} characters, seed={seed}", flush=True)
                try:
                    with torch.inference_mode():
                        waveform, sample_rate = tts.generate(language="Russian", prompt_text=reference["reference_text"],
                            prompt_audio=prompt, prompt_audio_sr=rate, text=text,
                            stop_threshold=threshold, n_timesteps=10, inference_cfg=2.0, seed=seed)
                finally:
                    handle.remove()
                if waveform.ndim != 2 or waveform.shape[0] != 1:
                    raise ValueError("Unexpected native waveform shape")
                samples = waveform[0].detach().float().cpu().numpy()
                levels = signal_levels(samples, sample_rate)
                # Preserve raw amplitudes as well as a PCM16 listening copy.
                # PCM16 saturation must not hide out-of-range native output.
                sf.write(output / f"{name}.float.wav", samples, sample_rate, subtype="FLOAT")
                sf.write(output / f"{name}.wav", samples, sample_rate, subtype="PCM_16")
                write_json(output / f"{name}.raw_levels.json", levels)
                write_json(output / f"{name}.stop_scores.json", scores)
                finish = "stop" if len(scores) > 6 and scores[-1] >= threshold else "max_patches"
                report["results"].append(dict(id=name, characters=len(text), seconds=levels["seconds"],
                    finish_reason=finish, generated_patches=len(scores)-(finish == "stop"),
                    raw_peak=levels["peak"], raw_full_scale_fraction=levels["fraction_at_or_above_full_scale"],
                    elapsed_seconds=time.monotonic()-started, audio=f"{name}.wav"))
                report["complete"] = len(report["results"]) == len(CASES)
                write_json(output / "report.json", report)
                print(f"[native-compare] {name}: {levels['seconds']:.2f}s; finish={finish}; raw peak={levels['peak']:.3f}", flush=True)
        finally:
            del tts
            torch.cuda.empty_cache()


def summarize(root):
    import soundfile as sf
    root = Path(root).resolve()
    settings = json.loads((root / "comparison_settings.json").read_text())
    native = json.loads((root / "native/report.json").read_text())
    if not native["complete"]:
        raise ValueError("Native generation is incomplete")
    rows, files = [], []
    for case in CASES:
        folder = root / "sglang" / case
        remote = json.loads((folder / "settings.json").read_text())
        if any(remote.get(key) != settings[key] for key in ("release_id", "reference_sha256", "seed", "stop_threshold")):
            raise ValueError(f"SGLang comparison settings changed: {case}")
        expected = (root / "texts" / f"{case}.txt").read_text().strip()
        for mode in ("full", "stream"):
            request = json.loads((folder / f"long_{mode}.request.json").read_text())
            expected_generation = {key: settings[key] for key in
                                   ("n_timesteps", "inference_cfg", "stop_threshold", "min_patches", "max_patches")}
            if (request["input"] != expected or request.get("language") != settings["language"]
                    or request.get("stage_params", {}).get("talker") != expected_generation):
                raise ValueError("Compared target texts or generation parameters differ")
        sg = {row["name"]: row for row in json.loads((folder / "summary.json").read_text())}
        paths = [("native", root / "native" / f"{case}.wav",
                  next(row for row in native["results"] if row["id"] == case).get("finish_reason"))]
        for mode in ("full", "stream"):
            row = sg[f"long_{mode}"]
            if "error" in row:
                raise ValueError(f"SGLang failed for {case}/{mode}: {row['error']}")
            paths.append((f"sglang_{mode}", folder / f"long_{mode}.wav", row.get("finish_reason")))
        for backend, path, finish in paths:
            samples, rate = sf.read(path, dtype="float64", always_2d=True)
            if samples.shape[1] != 1:
                raise ValueError("Unexpected channel count")
            levels = signal_levels(samples[:, 0], rate)
            levels_path = path.with_suffix(".levels.json")
            write_json(levels_path, levels)
            rows.append(dict(case=case, backend=backend, audio=str(path.relative_to(root)),
                             finish_reason=finish, seconds=levels["seconds"], peak=levels["peak"]))
            files.extend([path, levels_path])
        files.extend([root / "native" / f"{case}.float.wav", root / "native" / f"{case}.raw_levels.json"])
    write_json(root / "comparison.json", dict(settings=settings, results=rows))
    # Convenient sharing bundle: generated audio/levels only, no reference voice
    # recording, transcript or base64 reference embedded in request JSON files.
    with zipfile.ZipFile(root / "comparison-audio.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files + [root / "comparison.json"]:
            archive.write(path, path.relative_to(root))
    print(f"[native-compare] Results: {root}\n[native-compare] Audio archive: {root / 'comparison-audio.zip'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    native = commands.add_parser("native")
    native.add_argument("root")
    native.add_argument("--release", default="training/exports/FireRedTTS3-Base-Russian-20s-07")
    native.add_argument("--base", default="models/FireRedTTS3")
    native.add_argument("--seed", type=int, default=1234)
    native.add_argument("--stop-threshold", type=float, default=.5)
    commands.add_parser("summarize").add_argument("root")
    args = parser.parse_args()
    if args.command == "native":
        run_native(args.root, args.release, args.base, args.seed, args.stop_threshold)
    else:
        summarize(args.root)


if __name__ == "__main__":
    main()

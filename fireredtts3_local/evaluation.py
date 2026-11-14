"""Fixed text/reference benchmarks; independent speaker and ASR measurements."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

import numpy as np
import soundfile as sf
import torch
import torchaudio

from .training_data import file_digest, language_name, plain_text
from .training_run import isolated_rng


def word_error_rate(reference, hypothesis):
    def normalize(text):
        return re.findall(r"\w+", plain_text(text).lower().replace("ё", "е"))
    a, b = normalize(reference), normalize(hypothesis)
    if not a:
        raise ValueError("Empty evaluation reference text")
    previous = list(range(len(b)+1))
    for i, left in enumerate(a, 1):
        current = [i]
        for j, right in enumerate(b, 1):
            current.append(min(current[-1]+1, previous[j]+1, previous[j-1]+(left != right)))
        previous = current
    return previous[-1]/len(a)


def evaluate_benchmark(benchmark, output, adapter=None, speaker_model=None, asr_model=None):
    from .model_manager import ModelManager
    path = Path(benchmark).resolve()
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Benchmark requires nonempty unique case IDs")
    normalized = []
    for row in rows:
        ref = Path(row["reference_audio"])
        ref = ref if ref.is_absolute() else path.parent/ref
        if not ref.is_file() or not row["reference_text"].strip() or not row["text"].strip() or not row.get("speaker_id"):
            raise ValueError("Each case needs a reference recording/text, target text and speaker_id")
        normalized.append({**row, "reference_audio": str(ref.resolve()), "reference_sha256": file_digest(ref),
                           "language": language_name(row.get("language", "Russian")),
                           "seed": int(row.get("seed", 1234))})
    signature = hashlib.sha256(json.dumps(normalized, sort_keys=True).encode()).hexdigest()
    root = Path(output)
    if root.exists() and any(root.iterdir()):
        raise ValueError("Benchmark output must be new/empty")
    root.mkdir(parents=True, exist_ok=True)
    extractor = scorer = recognizer = None
    # Keep evaluation networks on CPU to avoid displacing TTS from GPU memory.
    if speaker_model:
        from transformers import AutoFeatureExtractor, WavLMForXVector
        extractor = AutoFeatureExtractor.from_pretrained(speaker_model)
        scorer = WavLMForXVector.from_pretrained(speaker_model).eval()
    if asr_model:
        from faster_whisper import WhisperModel
        recognizer = WhisperModel(asr_model, device="cpu", compute_type="int8")
    def embedding(audio, sr):
        x = torchaudio.functional.resample(audio.cpu(), sr, 16000).mean(0).numpy()
        inputs = extractor(x, sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            return torch.nn.functional.normalize(scorer(**inputs).embeddings, dim=-1)
    manager = ModelManager()
    results = []
    try:
        tts = manager.load_base(adapter_path=adapter, attention_backend="sdpa", torch_compile=False)
        for i, row in enumerate(normalized):
            ref, sr = torchaudio.load(row["reference_audio"])
            with isolated_rng(row["seed"]), torch.inference_mode():
                wav, rate = tts.generate(language=row["language"], prompt_text=plain_text(row["reference_text"]),
                                        prompt_audio=ref, prompt_audio_sr=sr, text=plain_text(row["text"]),
                                        seed=row["seed"], do_split=False)
            if wav.numel() == 0 or not torch.isfinite(wav).all():
                raise RuntimeError(f"Invalid generated waveform for {row['id']}")
            audio_path = root/f"case-{i:05d}.wav"
            sf.write(audio_path, wav.cpu().flatten().numpy(), rate)
            result = {"id": row["id"], "speaker_id": row["speaker_id"], "audio": audio_path.name,
                      "duration": wav.shape[-1]/rate}
            if scorer:
                result["speaker_similarity"] = float((embedding(ref, sr)*embedding(wav, rate)).sum())
            if recognizer:
                segments, _ = recognizer.transcribe(str(audio_path), language="ru" if row["language"] == "Russian" else None)
                transcript = " ".join(segment.text for segment in segments)
                result.update(asr_text=transcript, wer=word_error_rate(row["text"], transcript))
            results.append(result)
            report = {"benchmark_sha256": signature, "adapter": str(adapter) if adapter else None,
                      "speaker_model": speaker_model, "asr_model": asr_model,
                      "complete": len(results)==len(rows), "results": results}
            (root/"report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    finally:
        manager.unload()
    return root/"report.json"


def compare_reports(baseline, adapted, max_sim_drop=.02, max_wer_increase=.01):
    a, b = [json.loads(Path(p).read_text(encoding="utf-8")) for p in (baseline, adapted)]
    for field in ("benchmark_sha256", "speaker_model", "asr_model"):
        if a[field] != b[field]:
            raise ValueError(f"Benchmark/scorer mismatch: {field}")
    if not a.get("complete") or not b.get("complete"):
        raise ValueError("Cannot compare incomplete benchmark runs")
    if not a.get("speaker_model") or not a.get("asr_model"):
        raise ValueError("Automatic quality gate requires BOTH independent speaker and ASR scores")
    before = {r["id"]: r for r in a["results"]}
    after = {r["id"]: r for r in b["results"]}
    if not before or before.keys() != after.keys():
        raise ValueError("Case IDs differ")
    deltas = []
    speakers = {}
    for key, old in before.items():
        new = after[key]
        if old["speaker_id"] != new["speaker_id"]:
            raise ValueError("Speaker labels differ")
        sim = new["speaker_similarity"] - old["speaker_similarity"]
        wer = new["wer"] - old["wer"]
        if not np.isfinite([sim, wer]).all():
            raise ValueError("Non-finite benchmark scores")
        deltas.append((sim, wer))
        speakers.setdefault(old["speaker_id"], []).append(sim)
    mean_sim, mean_wer = np.mean(deltas, axis=0)
    worst_speaker = min(float(np.mean(v)) for v in speakers.values())
    return {"cases": len(deltas), "speakers": len(speakers),
            "mean_similarity_delta": float(mean_sim), "worst_speaker_mean_delta": worst_speaker,
            "mean_wer_delta": float(mean_wer),
            "automatic_gate_passed": bool(worst_speaker >= -max_sim_drop and mean_wer <= max_wer_increase),
            "thresholds": {"max_sim_drop": max_sim_drop, "max_wer_increase": max_wer_increase},
            "listening_required": "Check stress, prosody, endings and voice identity blind; these thresholds are engineering choices, not validated perceptual limits."}

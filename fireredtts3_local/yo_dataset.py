"""Bounded-memory YO-CPT-ru import from local Parquet shards or JSONL.

Only alignment-consistent transcripts are accepted. The legacy segmented mode
uses punctuation/pauses; complete-utterance mode keeps whole punctuated records.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import io
import json
import math
from pathlib import Path
import re

import numpy as np
import soundfile as sf
import torch
import torchaudio

from .training_data import file_digest, plain_text, write_manifest


def words(text):
    return re.findall(r"[\w]+", plain_text(text).casefold().replace("ё", "е"))


def parse_alignment(value):
    matches = list(re.finditer(r"([^\s]+)\((\d+(?:\.\d+)?),(\d+(?:\.\d+)?)\)", value))
    if not matches or re.sub(r"([^\s]+)\((\d+(?:\.\d+)?),(\d+(?:\.\d+)?)\)", "", value).strip():
        raise ValueError("invalid_alignment")
    result = []
    last = 0.0
    for match in matches:
        token, start, end = match.group(1), float(match.group(2)), float(match.group(3))
        if not math.isfinite(end) or end <= start or start < last - 0.025:
            raise ValueError("nonmonotonic_alignment")
        last = end
        if token.startswith("[pause"):
            continue
        token = plain_text(token.replace("[+]", "+"))
        if len(words(token)) != 1:
            raise ValueError("ambiguous_alignment_word")
        result.append({"word": token, "start": start, "end": end})
    if not result:
        raise ValueError("empty_alignment")
    return result


def has_sentence_ending(text):
    """Punctuation gate, not a claim of semantic completeness or ASR accuracy.

    Permit closing quotes/brackets and ?!/!!; reject ellipses, including spaced
    dots and combinations such as !.. or …?. Do not add missing punctuation.
    """
    tail = re.search(r'([.!?…⋯\s»”’"\'\)\]\}]+)$', text.strip())
    if not tail:
        return False
    marks = re.sub(r'[\s»”’"\'\)\]\}]', '', tail[1])
    return bool(marks) and all(c in '.!?' for c in marks) and marks.count('.') <= 1


def aligned_segments(row, max_seconds=10.0, min_seconds=1.5, pause_seconds=0.18,
                     segmentation="bounded"):
    if segmentation not in {"bounded", "complete-utterance"}:
        raise ValueError("Unknown segmentation mode")
    alignment = parse_alignment(row["text_alignment"])
    expected = [words(w["word"])[0] for w in alignment]
    if words(row["text"]) != expected:
        raise ValueError("alignment_transcript_mismatch")
    # Denormalized numbers may change words. Keep punctuated text only when it
    # is lexically identical to the aligned ASR transcript after stress removal.
    text = plain_text(row.get("text_denorm") or row["text"])
    if words(text) != expected:
        raise ValueError("denormalized_transcript_mismatch")
    if segmentation == "complete-utterance":
        if not has_sentence_ending(text):
            raise ValueError("incomplete_sentence_ending")
        duration = float(row["duration"])
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("invalid_duration")
        if alignment[-1]["end"] > duration + 0.025:
            raise ValueError("alignment_outside_audio")
        # Preserve the whole source waveform, pauses and text. No duration cap,
        # no concatenation of separate utterances, and no invented sentence end.
        return [(0.0, duration, text)]
    spans = list(re.finditer(r"\w+", text))
    boundaries = []
    for i, current in enumerate(alignment):
        tail = text[spans[i].end():spans[i+1].start()] if i+1 < len(spans) else text[spans[i].end():]
        next_start = alignment[i+1]["start"] if i+1 < len(alignment) else float(row["duration"])
        # Do not invent punctuation or split inside a word to meet a duration cap.
        if re.search(r"[.!?;,:…]", tail) or next_start-current["end"] >= pause_seconds or i == len(alignment)-1:
            boundaries.append(i)
    segments = []
    first = 0
    while first < len(alignment):
        start = max(0.0, alignment[first]["start"] - 0.04)
        # Largest available natural boundary within the limit.
        choices = [i for i in boundaries if i >= first and alignment[i]["end"] + 0.04 - start <= max_seconds]
        if not choices:
            raise ValueError("no_safe_segment_boundary")
        last = max(choices)
        end = min(float(row["duration"]), alignment[last]["end"] + 0.04)
        # Mid-gap bounds keep neighboring reference/target segments disjoint.
        if first:
            start = max(start, (alignment[first-1]["end"] + alignment[first]["start"]) / 2)
        if last+1 < len(alignment):
            end = min(end, (alignment[last]["end"] + alignment[last+1]["start"]) / 2)
        if end - start < min_seconds:
            raise ValueError("segment_too_short")
        begin_char = 0 if first == 0 else spans[first].start()
        end_char = spans[last+1].start() if last+1 < len(spans) else len(text)
        segments.append((start, end, text[begin_char:end_char].strip()))
        first = last + 1
    return segments


def iter_records(paths):
    for raw in paths:
        path = Path(raw)
        if path.suffix == ".jsonl":
            with path.open(encoding="utf-8") as stream:
                for line in stream:
                    if line.strip():
                        yield json.loads(line), path.parent
        elif path.suffix == ".parquet":
            try:
                import pyarrow.parquet as pq
            except ImportError as exc:
                raise RuntimeError("Parquet import needs pyarrow: uv sync --frozen --extra data") from exc
            for batch in pq.ParquetFile(path).iter_batches(batch_size=32):
                for row in batch.to_pylist():
                    yield row, path.parent
        else:
            raise ValueError(f"Expected .jsonl or .parquet: {path}")


def import_yo(paths, output, min_mos=4.0, max_seconds=10.0, max_hours=50.0,
              max_speaker_minutes=30.0, source_revision="", log=print,
              segmentation="bounded", *, import_state=None, finalize_pairs=True,
              allow_empty=False, excluded_utterances=None):
    if not source_revision.strip():
        raise ValueError("Record the YO-CPT-ru source revision for reproducibility")
    if segmentation not in {"bounded", "complete-utterance"}:
        raise ValueError("Unknown segmentation mode")
    if (not all(math.isfinite(x) for x in (min_mos, max_seconds, max_hours, max_speaker_minutes))
            or max_seconds <= 0 or max_hours < 0 or max_speaker_minutes <= 0):
        raise ValueError("Invalid import limits")
    hours_budget = max_hours * 3600 if max_hours else math.inf
    root = Path(output).resolve()
    if root.exists() and any(root.iterdir()):
        raise ValueError("Import output must be a new/empty directory")
    (root / "audio").mkdir(parents=True, exist_ok=True)
    state = import_state if import_state is not None else {}
    manifest, rejected = [], Counter()
    seconds = state.setdefault('seconds', Counter())
    seen = state.setdefault('seen', set())
    excluded_utterances = excluded_utterances or set()
    total = state.get('total', 0.)
    for row, base in iter_records(paths):
        if total >= hours_budget:
            break
        try:
            utt = str(row["utt_id"])
            if utt in seen:
                raise ValueError("duplicate_utterance")
            seen.add(utt)
            if utt in excluded_utterances:
                raise ValueError('previously_used_utterance')
            if row.get("lang") != "ru" or not math.isfinite(float(row["mos_score"])) or float(row["mos_score"]) < min_mos:
                raise ValueError("language_or_quality")
            speaker = str(row.get("local_spk_id") or "")
            if not speaker:
                raise ValueError("missing_speaker")
            duration = float(row["duration"])
            if (not math.isfinite(duration) or duration <= 0
                    or (segmentation == "bounded" and duration > 120)):
                raise ValueError("invalid_duration")
            if seconds[speaker] + duration > max_speaker_minutes * 60:
                raise ValueError("speaker_budget")
            segments = aligned_segments(row, max_seconds, segmentation=segmentation)
            audio = row["audio"]
            if isinstance(audio, dict) and audio.get("bytes") is not None:
                audio = io.BytesIO(audio["bytes"])
            else:
                audio = Path(audio["path"] if isinstance(audio, dict) else audio)
                audio = audio if audio.is_absolute() else base / audio
            signal, sr = sf.read(audio, dtype="float32", always_2d=True)
            if signal.shape[1] != 1 or not np.isfinite(signal).all():
                raise ValueError("invalid_waveform")
            if abs(len(signal)/sr - duration) > 0.15:
                raise ValueError("duration_mismatch")
            if (np.abs(signal) >= 0.999).mean() > 0.001:
                raise ValueError("clipping")
            if max(s[1] for s in segments) > len(signal)/sr + 0.025:
                raise ValueError("alignment_outside_audio")
            if segmentation == "complete-utterance":
                # Record actual sample duration, including a small allowed
                # metadata discrepancy. Never silently drop trailing samples.
                segments = [(0.0, len(signal)/sr, segments[0][2])]
        except (ValueError, KeyError, TypeError, RuntimeError) as exc:
            rejected[str(exc) if isinstance(exc, ValueError) else type(exc).__name__] += 1
            continue
        source = "YO-CPT-ru:" + speaker.split("__spk")[0]
        for index, (start, end, text) in enumerate(segments):
            if total + end - start > hours_budget:
                rejected["hours_budget"] += 1
                break
            clip = torch.from_numpy(signal[round(start*sr):round(end*sr), 0].copy())[None]
            if sr != 24000:
                clip = torchaudio.functional.resample(clip, sr, 24000)
            # IDs become filenames through a hash; untrusted metadata cannot escape output.
            import hashlib
            name = hashlib.sha256(f"{utt}:{index}".encode()).hexdigest()[:24]
            dest = root / "audio" / f"{name}.wav"
            sf.write(dest, clip[0].numpy(), 24000, subtype="PCM_16")
            manifest.append({"utt_id": f"yo:{utt}:{index}", "audio": f"audio/{name}.wav",
                             "text": text, "language": "Russian", "speaker_id": "yo:" + speaker,
                             "global_spk_id": "yo:" + str(row["global_spk_id"]) if row.get("global_spk_id") else None,
                             "source_id": source, "segment_source_id": "yo:" + utt, "start": start, "end": end,
                             "mos_score": float(row["mos_score"]), "source_revision": source_revision})
            total += end - start
            seconds[speaker] += end - start
    # A bounded shard selection can end with singleton speakers. Exclude them
    # explicitly and count the removal, instead of producing an untrainable import.
    speaker_hashes = defaultdict(set)
    for item in manifest:
        item["audio_sha256"] = file_digest(root / item["audio"])
        speaker_hashes[item["speaker_id"]].add(item["audio_sha256"])
    kept = []
    for item in manifest:
        if finalize_pairs and len(speaker_hashes[item["speaker_id"]]) < 2:
            rejected["segments_without_distinct_reference"] += 1
            (root / item["audio"]).unlink()
        else:
            kept.append(item)
    manifest = kept
    state['total'] = total
    total = sum(item["end"] - item["start"] for item in manifest)
    write_manifest(root / "manifest.jsonl", manifest)
    report = {"accepted_segments": len(manifest), "hours": total/3600, "local_speakers": len({r["speaker_id"] for r in manifest}),
              "rejections": dict(rejected), "source_revision": source_revision,
              "min_distillmos": min_mos,
              "segmentation": segmentation,
              "max_segment_seconds": max_seconds if segmentation == "bounded" else None,
              "max_hours": max_hours or None, "max_speaker_minutes": max_speaker_minutes,
              "duration_seconds": {"min": min((r['end']-r['start'] for r in manifest), default=0),
                                   "max": max((r['end']-r['start'] for r in manifest), default=0)},
              "note": "Automatic filtering; inspect audio, text and speaker identities before training. Input order affects capped subsets."}
    (root / "import_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    log(json.dumps(report, ensure_ascii=False))
    if not manifest and not allow_empty:
        raise ValueError("No usable aligned segments; see import_report.json")
    return root / "manifest.jsonl"

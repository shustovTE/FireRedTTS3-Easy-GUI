"""Conservative Dialogs tail review: transcript-matched Whisper timestamps + VAD.

An acoustic detector alone cannot identify the end of a sentence. A crop is
allowed only after all supplied words match a confident ASR prefix, and Silero
detects speech after the last matched word plus the requested safety margin.
Uncertain records stay unchanged and are explicitly listed for listening.
"""
from __future__ import annotations

import argparse
from collections import Counter
import importlib.metadata
import json
import math
from pathlib import Path

import soundfile as sf

from .training_data import SpeakerPairs, assert_disjoint, file_digest, write_manifest
from .yo_dataset import words, validate_tail_padding
from .yo_expansion import checked_rows, write_json
from .yo_retrain import signature

METHOD = "dialogs-asr-vad-tail-v1"
VAD_OPTIONS = dict(threshold=.5, min_speech_duration_ms=100,
                   min_silence_duration_ms=100, speech_pad_ms=0)
ASR_OPTIONS = dict(language="ru", word_timestamps=True, vad_filter=False,
                   temperature=0., beam_size=5, condition_on_previous_text=False)


def decide_tail(text, duration, recognized, speech_intervals, padding=.2):
    """Pure decision rule; timestamps and VAD intervals are in original seconds."""
    validate_tail_padding(padding)
    result = dict(action="review", reason="unmatched_transcript", cut_seconds=None)
    expected = words(text)
    tokens, ends, confidence, token_group_ends = [], [], [], []
    previous = 0.
    for word in recognized:
        start, end, probability = (float(word[k]) for k in ("start", "end", "probability"))
        if (not all(math.isfinite(v) for v in (start, end, probability))
                or not 0 <= start < end <= duration + .025 or start < previous - .025
                or not 0 <= probability <= 1):
            return dict(result, reason="invalid_word_timestamps")
        previous = end
        normalized = words(word["word"])
        tokens.extend(normalized)
        ends.extend([end] * len(normalized))
        confidence.extend([probability] * len(normalized))
        token_group_ends.extend([False] * max(0, len(normalized)-1) + ([True] if normalized else []))
    if not expected or tokens[:len(expected)] != expected:
        return result
    n = len(expected)
    if not token_group_ends[n-1]:
        return dict(result, reason="transcript_boundary_inside_word")
    last_end = ends[n-1]
    result.update(last_word_end=last_end, matched_words=n,
                  last_word_probability=confidence[n-1], mean_word_probability=sum(confidence[:n])/n)
    if confidence[n-1] < .8 or result["mean_word_probability"] < .8:
        return dict(result, reason="low_word_confidence")
    cut = min(duration, last_end + padding)
    tail_speech = 0.
    previous_end = 0.
    for start, end in speech_intervals:
        if (not all(math.isfinite(v) for v in (start, end)) or not 0 <= start < end <= duration + .025
                or start < previous_end):
            return dict(result, reason="invalid_acoustic_intervals")
        previous_end = end
        tail_speech += max(0., min(end, duration) - max(start, cut))
    result["speech_seconds_after_margin"] = tail_speech
    # One Silero frame is ~32 ms. A smaller overlap is not reliable evidence.
    if tail_speech < .032 or duration <= cut:
        return dict(result, action="keep", reason="no_detected_speech_after_margin")
    return dict(result, action="trim", reason="matched_text_and_detected_tail_speech", cut_seconds=cut)


class WhisperVadAnalyzer:
    def __init__(self, model="large-v3", device="cpu", compute_type="int8"):
        from faster_whisper.utils import download_model
        path = Path(model)
        self.path = path.resolve() if path.is_dir() else Path(download_model(model, local_files_only=True)).resolve()
        self.device, self.compute_type, self.model = device, compute_type, None
        files = [p for p in sorted(self.path.iterdir()) if p.is_file() and p.suffix in {".bin", ".json", ".txt"}]
        if not (self.path / "model.bin").is_file():
            raise ValueError("Expected a locally cached CTranslate2 Whisper model")
        self.identity = dict(model_path=str(self.path), files={p.name: file_digest(p) for p in files},
            faster_whisper=importlib.metadata.version("faster-whisper"),
            device=device, compute_type=compute_type, asr=ASR_OPTIONS, vad=VAD_OPTIONS)

    def analyze(self, row):
        from faster_whisper import WhisperModel
        from faster_whisper.audio import decode_audio
        from faster_whisper.vad import VadOptions, get_speech_timestamps
        if self.model is None:
            self.model = WhisperModel(str(self.path), device=self.device, compute_type=self.compute_type,
                                      cpu_threads=4, local_files_only=True)
        audio = decode_audio(row["audio"], sampling_rate=16000)
        segments, _ = self.model.transcribe(audio, **ASR_OPTIONS)
        recognized = [dict(word=w.word, start=w.start, end=w.end, probability=w.probability)
                      for segment in segments for w in (segment.words or [])]
        speech = get_speech_timestamps(audio, VadOptions(**VAD_OPTIONS), sampling_rate=16000)
        return dict(words=recognized, speech_intervals=[(r["start"]/16000, r["end"]/16000) for r in speech])


def review_dialogs(train="training/datasets/dialogs-neutral/train.jsonl",
                   validation="training/datasets/dialogs-neutral/validation.jsonl",
                   output="training/datasets/dialogs-tail-reviewed", padding=.2,
                   model="large-v3", device="cpu", compute_type="int8", *, analyzer=None, log=print):
    from filelock import FileLock
    validate_tail_padding(padding)
    if padding is None:
        raise ValueError("Tail review requires an explicit margin")
    paths = {"train": Path(train).resolve(), "validation": Path(validation).resolve()}
    root = Path(output).resolve()
    if any(p.is_relative_to(root) for p in paths.values()):
        raise ValueError("Review output must be separate from the original manifests")
    root.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(root.parent / f".{root.name}.review.lock"), timeout=0):
        analyzer = analyzer or WhisperVadAnalyzer(model, device, compute_type)
        contract = dict(method=METHOD, padding_seconds=padding, analyzer=analyzer.identity,
                        manifests={k: dict(path=str(p), sha256=file_digest(p)) for k,p in paths.items()})
        marker = root / "review_state.json"
        if root.exists() and any(root.iterdir()):
            if not marker.is_file() or json.loads(marker.read_text()).get("contract") != contract:
                raise ValueError("Review inputs/settings changed; use a new output directory")
            state = json.loads(marker.read_text())
        else:
            root.mkdir(parents=True, exist_ok=True)
            state = dict(contract=contract, receipts={}, complete=False)
        (root / "audio").mkdir(exist_ok=True)
        (root / "analysis").mkdir(exist_ok=True)
        state.update(status="running", complete=False)
        write_json(marker, state)
        try:
            groups = {k: checked_rows(p) for k,p in paths.items()}
            assert_disjoint(groups["train"], groups["validation"], "recording")
            all_original = groups["train"] + groups["validation"]
            if len({r["utt_id"] for r in all_original}) != len(all_original):
                raise ValueError("Duplicate Dialogs utterance across splits")
            if any("dialogs_tail_review" in r for r in all_original):
                raise ValueError("Review requires original Dialogs WAVs, not previously reviewed manifests")
            completed, audit = {}, []
            for split, rows in groups.items():
                prepared = []
                for index, row in enumerate(rows, 1):
                    key = signature(dict(utt_id=row["utt_id"], audio_sha256=row["audio_sha256"]))
                    receipt = root / "analysis" / f"{key}.json"
                    if key in state["receipts"]:
                        if not receipt.is_file() or file_digest(receipt) != state["receipts"][key]:
                            raise ValueError(f"Review receipt changed: {receipt}")
                        entry = json.loads(receipt.read_text())
                        updated = entry["row"]
                        if file_digest(updated["audio"]) != updated["audio_sha256"]:
                            raise ValueError(f"Reviewed WAV changed: {updated['audio']}")
                    else:
                        analysis = analyzer.analyze(row)
                        if file_digest(row["audio"]) != row["audio_sha256"]:
                            raise ValueError(f"Original Dialogs WAV changed during analysis: {row['audio']}")
                        decision = decide_tail(row["text"], row["duration"], analysis["words"],
                                               analysis["speech_intervals"], padding)
                        updated = dict(row)
                        if decision["action"] == "trim":
                            with sf.SoundFile(row["audio"]) as source:
                                if source.samplerate != 24000 or source.channels != 1 or source.subtype != "PCM_16":
                                    raise ValueError("Expected audited Dialogs 24 kHz mono PCM16 WAV")
                                frames = min(source.frames, math.ceil(decision["cut_seconds"]*source.samplerate))
                                dest = root / "audio" / f"{key}.wav"
                                partial = dest.with_suffix(".partial.wav")
                                sf.write(partial, source.read(frames, dtype="int16"), source.samplerate, subtype="PCM_16")
                                partial.replace(dest)
                                updated.update(audio=str(dest), audio_sha256=file_digest(dest),
                                               duration=frames/source.samplerate, end=frames/source.samplerate, start=0.)
                        updated["dialogs_tail_review"] = dict(method=METHOD, padding_seconds=padding,
                            original_audio_sha256=row["audio_sha256"], original_duration=row["duration"], **decision)
                        entry = dict(row=updated, analysis=analysis, original_audio=row["audio"], split=split)
                        write_json(receipt, entry)
                        state["receipts"][key] = file_digest(receipt)
                        write_json(marker, state)
                    prepared.append(updated)
                    audit.append(dict(utt_id=row["utt_id"], text=row["text"], split=split,
                        original_audio=row["audio"], reviewed_audio=updated["audio"],
                        removed_seconds=row["duration"]-updated["duration"], **updated["dialogs_tail_review"]))
                    log(f"[dialogs-tail] {split} {index}/{len(rows)}: {updated['dialogs_tail_review']['action']}")
                SpeakerPairs(prepared, 10., 10.)
                completed[split] = prepared
            assert_disjoint(completed["train"], completed["validation"], "recording")
            for split, rows in completed.items():
                write_manifest(root / f"{split}.jsonl", rows)
            write_json(root / "listening_audit.json", [r for r in audit if r["action"] != "keep"])
            counts = dict(Counter(r["action"] for r in audit))
            report = dict(method=METHOD, complete=True, counts=counts,
                          removed_seconds=sum(r["removed_seconds"] for r in audit),
                          padding_seconds=padding, uncertain_policy="keep original and list for manual review",
                          manifests={k: file_digest(root / f"{k}.jsonl") for k in completed})
            write_json(root / "audit_report.json", report)
            state.update(status="completed", complete=True)
            write_json(marker, state)
            log(f"[dialogs-tail] Complete: {counts}; listen to all proposed trims and uncertain cases in listening_audit.json")
            return report
        except BaseException as error:
            state.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                         complete=False, error_type=type(error).__name__, error=str(error))
            write_json(marker, state)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", default="training/datasets/dialogs-neutral/train.jsonl")
    parser.add_argument("--validation", default="training/datasets/dialogs-neutral/validation.jsonl")
    parser.add_argument("--output", default="training/datasets/dialogs-tail-reviewed")
    parser.add_argument("--tail-padding-seconds", type=float, default=.2)
    parser.add_argument("--asr-model", default="large-v3", help="Local CTranslate2 directory or already cached model")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--compute-type", default="int8")
    args = parser.parse_args()
    print(json.dumps(review_dialogs(args.train, args.validation, args.output, args.tail_padding_seconds,
        args.asr_model, args.device, args.compute_type), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

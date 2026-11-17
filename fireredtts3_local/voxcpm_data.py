"""Prepare an audited VoxCPM union from the three existing Russian selections.

Only edge VAD trimming is performed. Transcripts, speaker labels, original split
assignments and selection metadata are retained; originals are never modified.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
import hashlib
import json
import math
from pathlib import Path
import re
import tempfile
from types import SimpleNamespace

from filelock import FileLock
import soundfile as sf

from .training_data import dataset_fingerprint, file_digest, plain_text

SCHEMA = "voxcpm2-ru-data-v1"
DEFAULT_OUTPUT = "training/datasets/voxcpm2-ru"
SOURCE_SCHEMAS = {
    "dialogs": "dialogs-original-base-v1",
    "spontaneous": "cv-spontaneous-dialogs-v1",
    "scripted": "cv-scripted-27-v1",
}
SPLITS = ("train", "validation")
PADDING_SECONDS = 0.2
PREFERRED_REFERENCE_SECONDS = 30.0
VALIDATION_ORDER = "corpus-speaker-round-robin-v1"
# Kept here so dataset auditing need not import torch/torchaudio or a model.
# They must match the pinned detector in vad_trim; a test checks that contract.
VAD_VERSION = "6.2.3"
VAD_MODEL_SHA256 = "e1122837f4154c511485fe0b9c64455f7b929c96fbb8d79fbdb336383ebd3720"
VAD_WHEEL_SHA256 = "7b7f5436cfcb02fae583a05b512ea96467fd449fe54cb49a5e4f06c51a1e43b8"
VAD_SETTINGS = {"threshold": 0.5, "neg_threshold": 0.35, "min_speech_duration_ms": 0,
                "min_silence_duration_ms": 100, "speech_pad_ms": 0, "max_speech_duration_s": None,
                "sampling_rate": 16000, "return_seconds": False}


def _validate_vad(metadata):
    expected = {"package": "silero-vad", "version": VAD_VERSION, "backend": "torchscript", "device": "cpu",
                "model_resource": "silero_vad.data/silero_vad.jit", "model_sha256": VAD_MODEL_SHA256,
                "pypi_wheel_sha256": VAD_WHEEL_SHA256, "settings": VAD_SETTINGS}
    if not isinstance(metadata, dict) or any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError("Require pinned Silero VAD 6.2.3 model and settings in preparation metadata")


def _validate_trim(trim, source_info, output_info, vad):
    if trim.get("vad") != vad or trim.get("padding_seconds") != PADDING_SECONDS:
        raise ValueError("VAD metadata or 200ms padding changed")
    rate, frames = source_info.samplerate, source_info.frames
    start, end = trim.get("start_sample"), trim.get("end_sample")
    if (isinstance(start, bool) or isinstance(end, bool) or not isinstance(start, int)
            or not isinstance(end, int) or not 0 <= start < end <= frames):
        raise ValueError("Invalid VAD source sample bounds")
    if (trim.get("sample_rate") != rate or trim.get("original_samples") != frames
            or trim.get("output_samples") != end - start
            or trim.get("leading_removed_samples") != start
            or trim.get("trailing_removed_samples") != frames - end
            or trim.get("padding_samples") != math.floor(PADDING_SECONDS * rate)):
        raise ValueError("VAD sample accounting changed")
    speech = trim.get("speech_intervals")
    if not isinstance(speech, list) or not speech:
        raise ValueError("Missing VAD speech boundaries")
    previous = 0
    for segment in speech:
        a, b = segment.get("start"), segment.get("end")
        if (isinstance(a, bool) or isinstance(b, bool) or not isinstance(a, int) or not isinstance(b, int)
                or not previous <= a < b <= frames):
            raise ValueError("Invalid VAD speech intervals")
        previous = b
    padding = math.floor(PADDING_SECONDS * rate)
    if start != max(0, speech[0]["start"] - padding) or end != min(frames, speech[-1]["end"] + padding):
        raise ValueError("VAD trim does not preserve first-to-last speech with 200ms padding")
    if (output_info.samplerate != rate or output_info.frames != end - start
            or output_info.format != "WAV" or output_info.subtype != "FLOAT"
            or abs(trim["duration"] - output_info.duration) > 1 / rate):
        raise ValueError("VAD output format/duration differs from recorded sample slice")


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _write_rows(path, rows):
    with Path(path).open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def _read_rows(path):
    rows = []
    with Path(path).open(encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"Expected an object at {path}:{line_number}")
                rows.append(row)
    if not rows:
        raise ValueError(f"Empty manifest: {path}")
    return rows


def _words(text):
    return " ".join(re.findall(r"\w+", plain_text(text).casefold().replace("ё", "е")))


def _digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def _pcm_info(path):
    """Return sample identity and exact decoded dimensions (MP3 headers estimate)."""
    info = sf.info(path)
    digest = hashlib.sha256(json.dumps(["float32-le", info.samplerate, info.channels]).encode() + b"\0")
    if info.format in {"MP3", "MPEG"}:
        # SoundFile.blocks can pad to the estimated MP3 header length. The
        # whole-file decoder returns the actual samples, as does vad_trim.
        samples, rate = sf.read(path, dtype="float32", always_2d=True)
        if rate != info.samplerate:
            raise ValueError(f"Decoded sample rate differs from source metadata: {path}")
        blocks = (samples,)
    else:
        blocks = sf.blocks(path, blocksize=262144, dtype="float32", always_2d=True)
    frames = 0
    for block in blocks:
        digest.update(block.astype("<f4", copy=False).tobytes(order="C"))
        frames += len(block)
    if frames != info.frames and info.format not in {"MP3", "MPEG"}:
        raise ValueError(f"Decoded samples differ from lossless source metadata: {path}")
    decoded = SimpleNamespace(samplerate=info.samplerate, channels=info.channels, frames=frames,
                              duration=frames / info.samplerate, format=info.format, subtype=info.subtype)
    return digest.hexdigest(), decoded


def _pcm_digest(path):
    """Hash decoded samples, excluding WAV metadata such as PEAK timestamps."""
    return _pcm_info(path)[0]


def _describe(rows):
    actors = defaultdict(list)
    for row in rows:
        actors[row["speaker_id"]].append(row)
    return {
        "clips": len(rows),
        "hours": sum(row["duration"] for row in rows) / 3600,
        "speakers": len({row["speaker_id"] for row in rows}),
        "slots": dict(sorted(Counter(row.get("selection_slot", "unspecified") for row in rows).items())),
        "categories": dict(sorted(Counter(row.get("category", "unspecified") for row in rows).items())),
        "actors": {speaker: {"clips": len(clips), "seconds": sum(row["duration"] for row in clips),
                              "slots": dict(sorted(Counter(row.get("selection_slot", "unspecified")
                                                          for row in clips).items()))}
                   for speaker, clips in sorted(actors.items())},
    }


def _load_sources(paths):
    """Validate source reports and audio without importing any FireRed models."""
    records, signatures, cache = {}, {}, {}
    for dataset, root in paths.items():
        report_path = root / "report.json"
        report = _json(report_path)
        if report.get("schema") != SOURCE_SCHEMAS[dataset] or report.get("complete") is not True:
            raise ValueError(f"Require complete original {dataset} preparation ({SOURCE_SCHEMAS[dataset]})")
        signature = {"path": str(root), "report_sha256": file_digest(report_path), "manifests": {}}
        if "recipe_sha256" in report:
            actual = file_digest(root / "recipe.json")
            if actual != report["recipe_sha256"]:
                raise ValueError(f"Source {dataset} recipe changed")
            signature["recipe_sha256"] = actual
        records[dataset] = {}
        for split in SPLITS:
            manifest = root / f"{split}.jsonl"
            rows = _read_rows(manifest)
            expected = report.get("datasets", {}).get(split, {}).get("fingerprint")
            if not expected or dataset_fingerprint(rows) != expected:
                raise ValueError(f"Source {dataset}/{split} manifest fingerprint changed")
            signature["manifests"][split] = file_digest(manifest)
            signature["manifests"][split + "_fingerprint"] = expected
            seen, checked = set(), []
            for row in rows:
                for key in ("audio", "text", "speaker_id", "utt_id"):
                    if not isinstance(row.get(key), str) or not row[key].strip():
                        raise ValueError(f"Source {dataset}/{split} missing {key}")
                if row["utt_id"] in seen:
                    raise ValueError(f"Duplicate source utterance: {dataset}/{split}/{row['utt_id']}")
                seen.add(row["utt_id"])
                if not _words(row["text"]):
                    raise ValueError(f"Empty source transcript: {row['utt_id']}")
                if row.get("language", "Russian") not in {"Russian", "ru"}:
                    raise ValueError(f"Expected Russian source: {row['utt_id']}")
                audio = Path(row["audio"])
                audio = (audio if audio.is_absolute() else root / audio).resolve()
                if audio not in cache:
                    pcm, info = _pcm_info(audio)
                    if info.channels != 1 or not info.frames or not math.isfinite(info.duration):
                        raise ValueError(f"Source audio must be nonempty mono: {audio}")
                    cache[audio] = (file_digest(audio), float(info.duration), pcm, info.frames, info.samplerate)
                digest, duration, pcm, frames, rate = cache[audio]
                if row.get("audio_sha256") and row["audio_sha256"] != digest:
                    raise ValueError(f"Source audio hash changed: {audio}")
                recorded = row.get("duration")
                if (not isinstance(recorded, (int, float)) or not math.isfinite(recorded)
                        or recorded <= 0 or abs(recorded - duration) > 0.15):
                    raise ValueError(f"Source audio duration changed: {audio}")
                if dataset in {"dialogs", "spontaneous"}:
                    maximum = 10 if dataset == "dialogs" else 30
                    if (duration > maximum + 1e-6 or row.get("start", 0) != 0
                            or row.get("dialogs_tail_review") or row.get("tail_trim")):
                        raise ValueError(f"Require original unreviewed {dataset} selection: {row['utt_id']}")
                checked.append({**row, "audio": str(audio), "duration": duration,
                                "audio_sha256": digest, "_source_pcm_sha256": pcm,
                                "_source_samples": frames, "_source_sample_rate": rate,
                                "_dataset": dataset, "_split": split})
            records[dataset][split] = checked
            signature["manifests"][split + "_audio_fingerprint"] = _digest_json([
                [row["utt_id"], row["audio"], row["audio_sha256"], row["_source_pcm_sha256"], row["duration"]]
                for row in checked])
        signatures[dataset] = signature
    return records, signatures


def _identity_keys(row):
    keys = {"path:" + row["source_audio"], "original:" + row["source_sha256"],
            "trimmed:" + row["audio_sha256"], "pcm:" + row["pcm_sha256"],
            "pcm:" + row["source_pcm_sha256"]}
    for name in ("source_id", "segment_source_id"):
        if row.get(name):
            # Existing importers already namespace their source/session IDs.
            keys.add("source:" + str(row[name]))
    return keys


def _assert_no_leakage(train, validation):
    held_words = {_words(row["text"]) for row in validation}
    held_keys = {key for row in validation for key in _identity_keys(row)}
    if any(_words(row["text"]) in held_words for row in train):
        raise ValueError("Prepared train/validation normalized transcript leakage")
    if any(_identity_keys(row) & held_keys for row in train):
        raise ValueError("Prepared train/validation source or audio leakage")


def _distinct(target, ref):
    return (target["utt_id"] != ref["utt_id"]
            and target["source_audio"] != ref["source_audio"]
            and target["source_sha256"] != ref["source_sha256"]
            and target["source_pcm_sha256"] != ref["source_pcm_sha256"]
            and target["audio_sha256"] != ref["audio_sha256"]
            and target["pcm_sha256"] != ref["pcm_sha256"]
            and _words(target["text"]) != _words(ref["text"]))


def _reference(target, pool):
    eligible = [ref for ref in pool if _distinct(target, ref)]
    if not eligible:
        return None
    previous = target.get("ref_audio") or target.get("reference_audio")
    short = [ref for ref in eligible if ref["duration"] <= PREFERRED_REFERENCE_SECONDS]
    if previous and short:
        matches = [ref for ref in short if previous in {ref["source_audio"], ref["audio"]}]
        if matches:
            return matches[0]
    # Long references remain whole; 30 seconds is a preference, not a filter.
    if not short:
        return min(eligible, key=lambda row: (row["duration"], row["utt_id"]))
    short.sort(key=lambda row: row["utt_id"])
    position = int(hashlib.sha256(target["utt_id"].encode()).hexdigest()[:16], 16) % len(short)
    return short[position]


def _validation_order(pairs):
    """Cover corpora and available speakers early without dropping any target.

    The pinned upstream trainer checks only its first ten validation batches.
    This ordering broadens that prefix; it does not make it a full-holdout test.
    """
    grouped = defaultdict(lambda: defaultdict(list))
    for pair in pairs:
        row = pair[0]
        dataset = row.get("_dataset", row.get("dataset_id"))
        grouped[dataset][row["speaker_id"]].append(pair)
    corpora = deque()
    for dataset in sorted(grouped):
        speakers = deque(deque(sorted(rows, key=lambda pair: pair[0]["utt_id"]))
                         for _, rows in sorted(grouped[dataset].items()))
        corpora.append(speakers)
    ordered = []
    while corpora:
        speakers = corpora.popleft()
        samples = speakers.popleft()
        ordered.append(samples.popleft())
        if samples:
            speakers.append(samples)
        if speakers:
            corpora.append(speakers)
    return ordered


def _reject(row, reason, **extra):
    return {"dataset": row.get("_dataset", row.get("dataset_id")),
            "split": row.get("_split", row.get("split")), "utt_id": row["utt_id"],
            "speaker_id": row["speaker_id"], "reason": reason, **extra}


def _trim_audio(source, destination, detector):
    from .vad_trim import trim_audio
    return trim_audio(source, destination, detector=detector, padding_seconds=PADDING_SECONDS)


def _new_detector():
    from .vad_trim import SileroDetector
    return SileroDetector()


def _check_output_location(output, roots):
    for source in roots:
        if output == source or output.is_relative_to(source) or source.is_relative_to(output):
            raise ValueError("Prepared output must be separate from all source selections")


def prepare(dialogs="training/datasets/dialogs-base-08",
            spontaneous="training/datasets/cv-spontaneous-09",
            scripted="training/datasets/cv-scripted-27", output=DEFAULT_OUTPUT,
            *, detector=None, log=print):
    """Trim every selected target, reserve holdouts, then pair inside each split."""
    paths = {name: Path(path).resolve() for name, path in
             (("dialogs", dialogs), ("spontaneous", spontaneous), ("scripted", scripted))}
    output = Path(output).resolve()
    _check_output_location(output, paths.values())
    sources, signatures = _load_sources(paths)
    policy = {"padding_seconds": PADDING_SECONDS,
              "reference_preferred_seconds": PREFERRED_REFERENCE_SECONDS,
              "reference_max_seconds": None, "scripted_target_max_seconds": None,
              "validation_order": VALIDATION_ORDER}
    if output.exists():
        existing = audit(output)
        if (existing["sources"] != signatures or existing["policy"] != policy
                or (detector is not None and existing["vad"] != dict(getattr(detector, "metadata", {})))):
            raise ValueError("Preparation already exists for different inputs/settings; choose a new output")
        return existing
    output.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output) + ".prepare.lock", timeout=0):
        if output.exists():
            raise ValueError("Preparation already exists; retry audit")
        detector = detector if detector is not None else _new_detector()
        _validate_vad(getattr(detector, "metadata", None))
        with tempfile.TemporaryDirectory(prefix=".voxcpm-data-", dir=output.parent) as temporary:
            staging = Path(temporary) / "prepared"
            staging.mkdir()
            trimmed, rejections, trim_audits = {split: [] for split in SPLITS}, [], []
            total = sum(len(rows) for groups in sources.values() for rows in groups.values())
            completed = 0
            for dataset, groups in sources.items():
                for split, rows in groups.items():
                    folder = staging / "audio" / dataset / split
                    folder.mkdir(parents=True)
                    for row in rows:
                        identity = _digest_json([dataset, split, row["utt_id"], row["audio_sha256"]])
                        destination = folder / (identity + ".wav")
                        relative = destination.relative_to(staging)
                        try:
                            result = _trim_audio(row["audio"], destination, detector)
                        except ValueError as exc:
                            reason = getattr(exc, "reason", None)
                            if reason not in {"no_speech", "nonfinite_audio", "invalid_audio", "invalid_boundaries"}:
                                # Source changes or programming errors must never become silent data filtering.
                                raise
                            rejections.append(_reject(row, "vad_" + reason))
                            completed += 1
                            continue
                        if result["source_sha256"] != row["audio_sha256"]:
                            raise ValueError(f"Source changed during VAD: {row['audio']}")
                        if not destination.is_file() or result["output_sha256"] != file_digest(destination):
                            raise ValueError("VAD output hash mismatch")
                        info = sf.info(destination)
                        duration = float(info.duration)
                        if info.channels != 1 or not info.frames or not math.isfinite(duration):
                            raise ValueError("VAD produced invalid audio")
                        if abs(duration - result["duration"]) > 1 / info.samplerate:
                            raise ValueError("VAD output duration mismatch")
                        trim = {**result, "source": row["audio"], "destination": str(output / relative),
                                "dataset_id": dataset, "split": split, "utt_id": row["utt_id"]}
                        trim_audits.append(trim)
                        trimmed[split].append({**row, "source_audio": row["audio"],
                            "source_sha256": row["audio_sha256"], "audio": str(output / relative),
                            "source_pcm_sha256": row["_source_pcm_sha256"],
                            "audio_sha256": result["output_sha256"], "pcm_sha256": _pcm_digest(destination),
                            "duration": duration,
                            "_relative": relative.as_posix(), "_trim": trim})
                        completed += 1
                        if completed % 100 == 0 or completed == total:
                            log(f"[voxcpm-data] VAD {completed}/{total}; rejected={len(rejections)}")

            # Reserve the original holdouts first, including recording/session identities.
            held_words = {_words(row["text"]) for row in trimmed["validation"]}
            held_keys = {key for row in trimmed["validation"] for key in _identity_keys(row)}
            kept_train = []
            for row in trimmed["train"]:
                reasons = []
                if _words(row["text"]) in held_words:
                    reasons.append("normalized_text")
                if _identity_keys(row) & held_keys:
                    reasons.append("source_or_audio")
                if reasons:
                    rejections.append(_reject(row, "holdout_overlap", overlap=reasons))
                else:
                    kept_train.append(row)
            trimmed["train"] = kept_train
            prepared = {}
            for split in SPLITS:
                groups = defaultdict(list)
                for row in trimmed[split]:
                    groups[(row["_dataset"], row["speaker_id"])].append(row)
                paired = []
                for pool in groups.values():
                    for row in pool:
                        ref = _reference(row, pool)
                        if ref is None:
                            rejections.append(_reject(row, "no_distinct_same_speaker_reference"))
                        else:
                            paired.append((row, ref))
                if not paired:
                    counts = dict(Counter(row["reason"] for row in rejections if row["split"] == split))
                    raise ValueError(f"No usable {split} pairs remain; rejection counts: {counts}")
                paired.sort(key=lambda pair: (pair[0]["_dataset"], pair[0]["utt_id"]))
                prepared[split] = _validation_order(paired) if split == "validation" else paired
            selected = {split: [row for row, _ in prepared[split]] for split in SPLITS}
            for dataset in paths:
                for split in SPLITS:
                    if not any(row["_dataset"] == dataset for row in selected[split]):
                        counts = dict(Counter(row["reason"] for row in rejections
                                              if row["dataset"] == dataset and row["split"] == split))
                        raise ValueError(f"Required corpus {dataset}/{split} has no usable pairs; rejection counts: {counts}")
            _assert_no_leakage(selected["train"], selected["validation"])
            manifests, datasets, corpora = {}, {}, {}
            retained_audio = set()
            for split, pairs in prepared.items():
                official, provenance = [], []
                for row, ref in pairs:
                    official.append({"audio": row["audio"], "text": row["text"], "ref_audio": ref["audio"],
                                     "duration": row["duration"], "ref_duration": ref["duration"]})
                    metadata = {key: value for key, value in row.items() if not key.startswith("_")}
                    metadata.update(dataset_id=row["_dataset"], split=split, ref_utt_id=ref["utt_id"],
                                    ref_audio=ref["audio"], ref_sha256=ref["audio_sha256"],
                                    ref_pcm_sha256=ref["pcm_sha256"], ref_text=ref["text"],
                                    ref_duration=ref["duration"], trim=row["_trim"])
                    provenance.append(metadata)
                    retained_audio.update((row["_relative"], ref["_relative"]))
                for name, rows in ((f"{split}.jsonl", official), (f"{split}-provenance.jsonl", provenance)):
                    _write_rows(staging / name, rows)
                    manifests[name] = file_digest(staging / name)
                datasets[split] = {**_describe(selected[split]), "fingerprint": dataset_fingerprint(official),
                                  "long_reference_pairs": sum(ref["duration"] > PREFERRED_REFERENCE_SECONDS
                                                              for _, ref in pairs)}
            for dataset in paths:
                corpora[dataset] = {split: {
                    "source": _describe(sources[dataset][split]),
                    "retained": _describe([row for row in selected[split] if row["_dataset"] == dataset]),
                } for split in SPLITS}
                folder = staging / "corpora" / dataset
                folder.mkdir(parents=True)
                for split in SPLITS:
                    rows = [row for row in _read_rows(staging / f"{split}-provenance.jsonl")
                            if row["dataset_id"] == dataset]
                    name = f"corpora/{dataset}/{split}.jsonl"
                    _write_rows(staging / name, rows)
                    manifests[name] = file_digest(staging / name)
            # Discard only new, unused derivatives, never any original/source file.
            for audio in (staging / "audio").rglob("*.wav"):
                if audio.relative_to(staging).as_posix() not in retained_audio:
                    audio.unlink()
            for name, rows in (("vad-audit.jsonl", trim_audits), ("rejections.jsonl", rejections)):
                _write_rows(staging / name, rows)
                manifests[name] = file_digest(staging / name)
            report = {
                "schema": SCHEMA, "complete": True, "sources": signatures, "policy": policy,
                "vad": dict(getattr(detector, "metadata", {})), "datasets": datasets, "corpora": corpora,
                "manifests": manifests, "rejections": dict(sorted(Counter(row["reason"] for row in rejections).items())),
                "source_clips": total, "vad_successful_clips": len(trim_audits),
                "audio": "VAD first-to-last speech with 200ms edge padding; internal pauses retained; no ASR or gain change",
                "selection": "existing selections and splits; no oversampling; holdouts win overlap conflicts",
                "validation": "known speakers may occur in both splits; disjoint source recordings, audio hashes and normalized word sequences",
                "audio_identity": "File SHA256 plus canonical decoded float32 PCM SHA256 before and after VAD; container headers excluded from PCM identity",
                "validation_order_note": "Round-robin across corpora and speakers, every target once. The pinned upstream trainer evaluates only its first 10 batches, not the complete held-out manifest.",
                "reference": "same corpus, speaker and split; distinct audio/text; prefer <=30s, otherwise shortest complete reference",
            }
            _write_json(staging / "report.json", report)
            # Recheck original audio and metadata before installing the complete directory.
            _, after = _load_sources(paths)
            if after != signatures:
                raise ValueError("Sources changed during preparation")
            staging.rename(output)
    return audit(output)


def audit(output=DEFAULT_OUTPUT):
    """Verify inputs, derivative hashes, references and union-level split leakage."""
    output = Path(output).resolve()
    report = _json(output / "report.json")
    if report.get("schema") != SCHEMA or report.get("complete") is not True:
        raise ValueError("Require a complete VoxCPM preparation")
    _validate_vad(report.get("vad"))
    if report.get("policy") != {"padding_seconds": PADDING_SECONDS,
                               "reference_preferred_seconds": PREFERRED_REFERENCE_SECONDS,
                               "reference_max_seconds": None, "scripted_target_max_seconds": None,
                               "validation_order": VALIDATION_ORDER}:
        raise ValueError("Prepared policy changed")
    paths = {dataset: Path(spec["path"]).resolve() for dataset, spec in report["sources"].items()}
    if set(paths) != set(SOURCE_SCHEMAS):
        raise ValueError("Expected all three source corpora")
    _check_output_location(output, paths.values())
    sources, signatures = _load_sources(paths)
    if signatures != report["sources"]:
        raise ValueError("Source manifests/reports changed after preparation")
    required = {"vad-audit.jsonl", "rejections.jsonl"}
    required.update(f"{split}{suffix}.jsonl" for split in SPLITS for suffix in ("", "-provenance"))
    required.update(f"corpora/{dataset}/{split}.jsonl" for dataset in SOURCE_SCHEMAS for split in SPLITS)
    if set(report["manifests"]) != required:
        raise ValueError("Prepared manifest audit is incomplete")
    for name, expected in report["manifests"].items():
        path = output / name
        if not path.resolve().is_relative_to(output) or file_digest(path) != expected:
            raise ValueError(f"Prepared file changed: {name}")
    source_lookup = {(dataset, split, row["utt_id"]): row
                     for dataset, groups in sources.items() for split, rows in groups.items() for row in rows}
    data, audio_cache = {}, {}
    for split in SPLITS:
        rows = _read_rows(output / f"{split}.jsonl")
        provenance = _read_rows(output / f"{split}-provenance.jsonl")
        if len(rows) != len(provenance) or dataset_fingerprint(rows) != report["datasets"][split]["fingerprint"]:
            raise ValueError(f"Prepared {split} fingerprint/metadata mismatch")
        lookup = {}
        for row, meta in zip(rows, provenance):
            if set(row) != {"audio", "text", "ref_audio", "duration", "ref_duration"}:
                raise ValueError("Unexpected official VoxCPM manifest fields")
            key = (meta["dataset_id"], split, meta["utt_id"])
            if key in lookup or key not in source_lookup:
                raise ValueError("Missing or repeated source provenance")
            lookup[key] = meta
            original = source_lookup[key]
            if (meta["split"] != split or meta["speaker_id"] != original["speaker_id"]
                    or meta["source_audio"] != original["audio"] or meta["source_sha256"] != original["audio_sha256"]
                    or meta["source_pcm_sha256"] != original["_source_pcm_sha256"]
                    or meta["text"] != original["text"]):
                raise ValueError("Source identity/text differs from prepared provenance")
            for field in row:
                if row[field] != meta[field]:
                    raise ValueError(f"Official manifest differs from provenance: {field}")
            for field, expected_hash, expected_pcm, duration in (
                    ("audio", meta["audio_sha256"], meta["pcm_sha256"], row["duration"]),
                    ("ref_audio", meta["ref_sha256"], meta["ref_pcm_sha256"], row["ref_duration"])):
                path = Path(row[field]).resolve()
                if not path.is_relative_to(output / "audio"):
                    raise ValueError("Prepared audio must remain inside its output")
                if path not in audio_cache:
                    audio_cache[path] = (file_digest(path), sf.info(path), _pcm_digest(path))
                actual, info, pcm = audio_cache[path]
                if (actual != expected_hash or pcm != expected_pcm or info.channels != 1 or not info.frames
                        or not math.isfinite(duration) or abs(info.duration - duration) > 1 / info.samplerate):
                    raise ValueError(f"Prepared audio changed: {path}")
            trim = meta["trim"]
            if (trim["source_sha256"] != meta["source_sha256"]
                    or trim["output_sha256"] != meta["audio_sha256"]
                    or trim["source"] != meta["source_audio"] or trim["destination"] != meta["audio"]):
                raise ValueError("VAD provenance changed")
            decoded_source = SimpleNamespace(frames=original["_source_samples"],
                                             samplerate=original["_source_sample_rate"])
            _validate_trim(trim, decoded_source,
                           audio_cache[Path(meta["audio"]).resolve()][1], report["vad"])
        for meta in provenance:
            ref = lookup.get((meta["dataset_id"], split, meta["ref_utt_id"]))
            if (ref is None or ref["speaker_id"] != meta["speaker_id"]
                    or ref["audio"] != meta["ref_audio"] or ref["audio_sha256"] != meta["ref_sha256"]
                    or ref["pcm_sha256"] != meta["ref_pcm_sha256"]
                    or ref["text"] != meta["ref_text"] or not _distinct(meta, ref)):
                raise ValueError("Reference must be distinct audio/text from the same speaker/corpus/split")
        data[split] = provenance
        if split == "validation":
            expected_order = [row[0]["audio"] for row in _validation_order([(row, None) for row in provenance])]
            if expected_order != [row["audio"] for row in provenance]:
                raise ValueError("Validation must retain deterministic corpus/speaker interleaving")
        if {row["dataset_id"] for row in provenance} != set(SOURCE_SCHEMAS):
            raise ValueError(f"Every required corpus must have usable {split} pairs")
        if _describe(provenance) != {key: report["datasets"][split][key] for key in _describe(provenance)}:
            raise ValueError(f"Prepared {split} statistics changed")
    _assert_no_leakage(data["train"], data["validation"])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare")
    p.add_argument("--dialogs", default="training/datasets/dialogs-base-08")
    p.add_argument("--spontaneous", default="training/datasets/cv-spontaneous-09")
    p.add_argument("--scripted", default="training/datasets/cv-scripted-27")
    p.add_argument("--output", default=DEFAULT_OUTPUT)
    p = commands.add_parser("audit")
    p.add_argument("output", nargs="?", default=DEFAULT_OUTPUT)
    args = vars(parser.parse_args())
    command = args.pop("command")
    print(json.dumps({"prepare": prepare, "audit": audit}[command](**args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

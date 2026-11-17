"""Prepare uncapped neutral Dialogs with a confirmed Russian Common Voice corpus."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import tempfile

from filelock import FileLock
import numpy as np
import soundfile as sf

from . import common_voice_download as sps_archive
from . import common_voice_scripted as cv
from . import dialogs_dataset as dialogs
from .training_data import dataset_fingerprint, file_digest
from .voxcpm_data import (FAIR_OUTPUT, FAIR_ROOT, FAIR_SOURCES, RAW_OUTPUT, RAW_ROOT, RAW_SOURCES, SPLITS,
                          _words, _write_json, _write_rows, prepare_raw_30s, prepare_spontaneous_1000)


def heldout_sessions(rows):
    families = defaultdict(Counter)
    for row in rows:
        session = row["session"]
        families[session.rsplit("_part", 1)[0]][session] += row["duration"]
    held = set()
    for sessions in families.values():
        if len(sessions) < 2:
            continue
        target = sum(sessions.values()) / 10
        remaining = sorted(sessions, key=lambda s: hashlib.sha256(("1234:" + s).encode()).digest())
        seconds = 0.
        while len(remaining) > 1:
            closest = min(remaining, key=lambda s: abs(target - seconds - sessions[s]))
            if seconds and abs(target - seconds - sessions[closest]) >= abs(target - seconds):
                break
            held.add(closest)
            seconds += sessions[closest]
            remaining.remove(closest)
    return held


def dialog_candidates(source):
    revision = dialogs.REVISION
    dialogs.fetch_files(source, dialogs.METADATA, revision)
    inventory_path = source / f"file-inventory-{revision}.json"
    files, inventory_hash = dialogs.file_inventory(source, revision, refresh=not inventory_path.exists())
    candidates, rejected = dialogs.eligible(dialogs.read_csv(source / "metadata.csv"), float("inf"), files)
    complete = []
    rejected = Counter(rejected)
    for row in candidates:
        try:
            row["text"] = cv.transcript(row["text"])
        except ValueError as exc:
            rejected[str(exc)] += 1
            continue
        complete.append(row)
    candidates = complete
    dialogs.fetch_files(source, [r["audio_path"] for r in candidates], revision)
    held = heldout_sessions(candidates)
    splits = {name: [] for name in SPLITS}
    for row in candidates:
        name = row["audio_path"]
        split = "validation" if row["session"] in held else "train"
        splits[split].append(dict(row, audio=str(source / name), utt_id="dialogs:" + name,
                                  speaker_id="dialogs:" + row["speaker_id"], language="Russian",
                                  source_id="dialogs:" + row["session"], segment_source_id="dialogs:" + name))
    return splits, {"repo": dialogs.REPO, "revision": revision, "metadata_sha256": file_digest(source / "metadata.csv"),
                    "inventory_sha256": inventory_hash, "heldout_sessions": sorted(held),
                    "metadata_rejections": rejected}


def assign_common_voice(rows, dialog_splits):
    # Keep every occurrence of a sentence on one side, also across corpora.
    held = {_words(r["text"]) for r in dialog_splits["validation"]}
    train = {_words(r["text"]) for r in dialog_splits["train"]}
    splits = {split: [] for split in SPLITS}
    for row in rows:
        words = _words(row["text"])
        is_validation = words in held or (words not in train and
                         int.from_bytes(hashlib.sha256(("1234:" + words).encode()).digest()[:8], "big") % 10 == 0)
        splits["validation" if is_validation else "train"].append(row)
    return splits


def save_source(dataset, folder, splits, details, *, installed_dir=None):
    folder.mkdir(parents=True, exist_ok=True)
    rejected, datasets = Counter(), {}
    seen_audio = set()
    for split in ("validation", "train"):
        accepted = []
        for row in splits[split]:
            path = folder / "audio" / row["cv_source_path"] if dataset == "scripted" else Path(row["audio"])
            if not path.is_file():
                rejected["missing_audio"] += 1
                continue
            try:
                samples, rate = sf.read(path, dtype="float32", always_2d=True)
            except sf.LibsndfileError:
                rejected["undecodable"] += 1
                continue
            if samples.shape[1] != 1 or not samples.size or not np.isfinite(samples).all() or not samples.any():
                rejected["invalid_audio"] += 1
                continue
            duration = len(samples) / rate
            if dataset == "dialogs":
                if abs(row["duration"] - duration) > .15:
                    rejected["duration_mismatch"] += 1
                    continue
                if (np.abs(samples) >= .999).mean() > .001:
                    rejected["clipping"] += 1
                    continue
            sha = file_digest(path)
            if sha in seen_audio:
                rejected["duplicate_audio"] += 1
                continue
            seen_audio.add(sha)
            final = installed_dir / "audio" / path.name if installed_dir else path
            accepted.append(dict(row, audio=str(final), duration=duration, audio_sha256=sha))
        _write_rows(folder / f"{split}.jsonl", accepted)
        datasets[split] = {"clips": len(accepted), "fingerprint": dataset_fingerprint(accepted)}
    _write_json(folder / "report.json", dict(details, schema=(RAW_SOURCES | FAIR_SOURCES)[dataset], complete=True,
                                             datasets=datasets, audio_rejections=dict(rejected)))


def import_spontaneous(archive, staging, output, dialog_splits):
    from .common_voice_spontaneous import collect
    original = sps_archive.extract_verified(archive, staging.parent / "original")
    folder = staging / "spontaneous"
    rows, excluded = collect(original, folder / "audio", original_for_vad=True)
    candidates = []
    rejections = Counter(r["reason"] for r in excluded)
    for row in rows:
        try:
            cv.transcript(row["text"])
        except ValueError as exc:
            rejections[str(exc)] += 1
            continue
        candidates.append(row)
    splits = assign_common_voice(candidates, dialog_splits)
    save_source("spontaneous", folder, splits,
                {"dataset_id": sps_archive.DATASET_ID, "raw_input": str(archive),
                 "archive_sha256": sps_archive.ARCHIVE_SHA256, "source_rejections": dict(rejections)},
                installed_dir=output / "spontaneous")


def import_sources(source, archive, output, log=print, *, spontaneous=False):
    source, archive, output = (Path(p).resolve() for p in (source, archive, output))
    output.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output) + ".lock", timeout=0):
        if output.exists():
            # prepare_raw_30s rechecks both source reports, manifests and audio.
            for dataset, schema in (FAIR_SOURCES if spontaneous else RAW_SOURCES).items():
                report = json.loads((output / dataset / "report.json").read_text())
                raw_input = source if dataset == "dialogs" else archive
                if (report.get("schema") != schema or report.get("complete") is not True
                        or report.get("raw_input") != str(raw_input)):
                    raise ValueError(f"Incomplete original source import at {output}")
            return
        (sps_archive if spontaneous else cv).verify_archive(archive)
        with tempfile.TemporaryDirectory(dir=output.parent, prefix=".voxcpm-originals-") as temporary:
            staging = Path(temporary) / "data"
            log("[sources] Reading the pinned Dialogs inventory and fetching eligible neutral recordings", flush=True)
            dialog_splits, details = dialog_candidates(source)
            save_source("dialogs", staging / "dialogs", dialog_splits, dict(details, raw_input=str(source)))
            if spontaneous:
                log("[sources] Reading confirmed Russian Spontaneous from original MP3s", flush=True)
                import_spontaneous(archive, staging, output, dialog_splits)
                staging.rename(output)
                return
            folder = staging / "scripted"
            log("[sources] Reading Russian Scripted validated recordings and confirmed sentences", flush=True)
            metadata = cv.extract_metadata(archive, folder / "metadata")
            rows, excluded, hashes = cv.candidates(metadata)
            log(f"[sources] Extracting {len(rows)} eligible Scripted recordings", flush=True)
            cv._extract_audio(archive, folder / "audio", rows, print)
            splits = assign_common_voice(rows, dialog_splits)
            save_source("scripted", folder, splits,
                        {"dataset_id": cv.DATASET_ID, "raw_input": str(archive), "archive_sha256": cv.ARCHIVE_SHA256, "metadata_sha256": hashes,
                         "metadata_rejections": dict(Counter(r["reason"] for r in excluded))},
                        installed_dir=output / "scripted")
            staging.rename(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dialogs", default="training/raw/dialogs")
    parser.add_argument("--spontaneous", action="store_true", help="Dialogs + confirmed Spontaneous, fixed 1000 updates")
    parser.add_argument("--archive")
    parser.add_argument("--sources")
    parser.add_argument("--output")
    args = parser.parse_args()
    sources = args.sources or (FAIR_ROOT if args.spontaneous else RAW_ROOT)
    output = args.output or (FAIR_OUTPUT if args.spontaneous else RAW_OUTPUT)
    archive = args.archive or (sps_archive.DEFAULT_ARCHIVE if args.spontaneous else cv.DEFAULT_ARCHIVE)
    import_sources(args.dialogs, archive, sources, spontaneous=args.spontaneous)
    report = (prepare_spontaneous_1000 if args.spontaneous else prepare_raw_30s)(sources, output)
    for name, groups in report["corpora"].items():
        for split, group in groups.items():
            stats = group["retained"]
            print(f"{name}/{split}: {stats['clips']} clips, {stats['hours']:.2f} hours")
    balance = report["balance"]
    highest = (balance["max_draws_per_clip"] if args.spontaneous else
               max(b["max_draws_per_clip"] for b in balance["buckets"].values()))
    print(f"Balanced epoch: {balance['unique_targets']} unique targets, {balance['target_draws']} draws, "
          f"{balance['question_fraction']:.2%} containing '?'. "
          f"Maximum draws for one clip: {highest}")
    print(f"Rejections: {report['rejections']}")


if __name__ == "__main__":
    main()

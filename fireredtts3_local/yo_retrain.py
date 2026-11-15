"""Reuse audited local YO WAVs, trim aligned tails, and prepare a fresh LoRA run.

No downloads, re-splitting, silence detection or text changes. Parquet reads
request metadata columns only. Completed metadata parts and WAVs are resumable.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import shutil

import soundfile as sf

from .training_data import (SpeakerPairs, assert_disjoint, dataset_fingerprint,
                            file_digest, write_manifest)
from .training_run import TrainingOptions, prepare_run_data, primary_examples_per_update
from .yo_dataset import aligned_segments, parse_alignment, validate_tail_padding
from .yo_expansion import REPO, checked_rows, check_shard, write_json
from .yo_training import clean_dialogs_holdout

SCHEMA = "yo-aligned-tail-v1"
COLUMNS = ("utt_id", "text", "text_denorm", "text_alignment", "duration", "local_spk_id")


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def metadata_records(path):
    import pyarrow.parquet as pq
    parquet = pq.ParquetFile(path)
    if not set(COLUMNS) <= set(parquet.schema_arrow.names):
        raise ValueError(f"Missing YO alignment columns: {path}")
    for batch in parquet.iter_batches(batch_size=1024, columns=list(COLUMNS)):
        yield from batch.to_pylist()


def raw_id(row):
    source = row.get("segment_source_id", "")
    if (not source.startswith("yo:") or row.get("utt_id") != source + ":0"
            or row.get("start") != 0 or "tail_trim" in row):
        raise ValueError(f"Expected original whole YO utterance, not a segment/previous crop: {row['utt_id']}")
    return source[3:]


def collect_alignments(plan, wanted, root, state, log):
    """Verify the pinned shards and checkpoint only selected metadata, no audio."""
    parts = root / "alignments"
    parts.mkdir(exist_ok=True)
    found = {}
    for index, entry in enumerate(plan["selected"], 1):
        # check_shard also validates the relative filename before using it.
        from .yo_expansion import validate_shard
        validate_shard(entry)
        path = parts / (Path(entry["path"]).stem + ".json")
        cached_hash = state["alignment_parts"].get(entry["path"])
        if cached_hash:
            if not path.is_file() or file_digest(path) != cached_hash:
                raise ValueError(f"Cached alignment metadata changed: {path}")
            records = json.loads(path.read_text(encoding="utf-8"))
        else:
            source = check_shard(plan["source_dir"], entry)
            records = [row for row in metadata_records(source) if str(row["utt_id"]) in wanted]
            write_json(path, records)
            state["alignment_parts"][entry["path"]] = file_digest(path)
            write_json(root / "preparation_state.json", state)
        for record in records:
            key = str(record["utt_id"])
            if key not in wanted or key in found:
                raise ValueError(f"Unexpected/duplicate raw utterance in alignment metadata: {key}")
            found[key] = (record, entry["path"])
        log(f"[yo-retrain] Alignment {index}/{len(plan['selected'])}; matched={len(found)}/{len(wanted)}")
    missing = wanted - found.keys()
    if missing:
        raise ValueError(f"Missing text_alignment for {len(missing)} selected WAVs; first: {sorted(missing)[:5]}")
    return found


def crop_row(row, record, shard, root, padding):
    """Keep every sample through ceil((last lexical word end + margin) * rate)."""
    if file_digest(row["audio"]) != row["audio_sha256"]:
        raise ValueError(f"Original WAV changed during preparation: {row['audio']}")
    if row["speaker_id"] != "yo:" + str(record["local_spk_id"]):
        raise ValueError(f"Alignment speaker differs: {row['utt_id']}")
    if abs(float(record["duration"]) - row["duration"]) > .15:
        raise ValueError(f"Whole WAV/raw duration differs: {row['utt_id']}")
    _, end, text = aligned_segments(dict(record, duration=row["duration"]),
        segmentation="complete-utterance", tail_padding_seconds=padding)[0]
    if text != row["text"]:
        raise ValueError(f"Alignment transcript differs: {row['utt_id']}")
    last = parse_alignment(record["text_alignment"])[-1]
    with sf.SoundFile(row["audio"]) as source:
        if source.samplerate != 24000 or source.channels != 1 or source.subtype != "PCM_16":
            raise ValueError("Expected audited 24 kHz mono PCM16 YO WAV")
        frames = min(source.frames, math.ceil(end * source.samplerate))
        if frames <= 0:
            raise ValueError("Empty aligned crop")
        duration = frames / source.samplerate
        trim = dict(method=SCHEMA, padding_seconds=padding, last_word=last["word"],
                    last_word_end=last["end"], original_duration=row["duration"],
                    removed_seconds=row["duration"] - duration,
                    original_audio_sha256=row["audio_sha256"], source_shard=shard,
                    alignment_sha256=hashlib.sha256(record["text_alignment"].encode()).hexdigest())
        name = signature(dict(utt_id=row["utt_id"], trim=trim))
        destination = root / "audio" / f"{name}.wav"
        receipt_path = destination.with_suffix(".json")
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
            if (receipt.get("tail_trim") != trim or not destination.is_file()
                    or file_digest(destination) != receipt.get("audio_sha256")):
                raise ValueError(f"Prepared audio/receipt changed: {destination}")
            digest = receipt["audio_sha256"]
        else:
            partial = destination.with_suffix(".partial.wav")
            if frames == source.frames:
                shutil.copyfile(row["audio"], partial)
            else:
                sf.write(partial, source.read(frames, dtype="int16"), source.samplerate, subtype="PCM_16")
            partial.replace(destination)
            digest = file_digest(destination)
            write_json(receipt_path, dict(tail_trim=trim, audio_sha256=digest))
    return {**row, "audio": str(destination), "audio_sha256": digest,
            "start": 0., "end": duration, "duration": duration, "tail_trim": trim}


def dataset_summary(rows):
    return dict(clips=len(rows), hours=sum(r["duration"] for r in rows)/3600,
                local_speakers=len({r["speaker_id"] for r in rows}),
                fingerprint=dataset_fingerprint(rows))


def prepare_retraining(expansion="training/datasets/yo-half", plan_path="training/plans/yo-half/plan.json",
                       output="training/datasets/yo-trimmed-06", padding=.2,
                       recipe_path="recipes/russian_trimmed_base_96gb.json", log=print):
    from filelock import FileLock
    validate_tail_padding(padding)
    if padding is None:
        raise ValueError("Retraining preparation requires an explicit tail margin")
    expansion, root, plan_path = Path(expansion).resolve(), Path(output).resolve(), Path(plan_path).resolve()
    plan = json.loads(plan_path.read_text())
    expansion_state = json.loads((expansion / "expansion_state.json").read_text())
    imported = json.loads((expansion / "split_report.json").read_text())
    plan_hash = file_digest(plan_path)
    if (plan.get("schema") != 1 or plan.get("repo") != REPO
            or not expansion_state.get("complete")
            or expansion_state.get("plan_sha256") != plan_hash or imported.get("plan_sha256") != plan_hash
            or imported.get("source_revision") != plan["revision"]
            or plan.get("filters", {}).get("segmentation") != "complete-utterance"):
        raise ValueError("Expected a completed whole-utterance YO expansion bound to this plan")
    if not plan["selected"] or len({r["path"] for r in plan["selected"]}) != len(plan["selected"]):
        raise ValueError("Empty/duplicate Parquet inventory")
    config = json.loads(Path(recipe_path).read_text())
    if config.get("initial_adapter") or config.get("resume_checkpoint"):
        raise ValueError("Fresh retraining recipe must not contain an initial adapter or resume checkpoint")
    options = TrainingOptions.parse(config["options"])
    if (options.sampling_strategy != "coverage_replay" or config["length_mode"] != "Epochs"
            or options.validation_scope != "speaker" or not options.expected_base_fingerprint):
        raise ValueError("Fresh recipe requires coverage_replay, Epochs, speaker holdout and a pinned base fingerprint")
    run = Path(config["output_dir"]).resolve()
    if run.exists() and any(run.iterdir()):
        raise ValueError("Training output is already populated; prepare a new run or explicitly resume its saved recipe")
    paths = {"train": expansion / "train.jsonl", "validation": expansion / "validation.jsonl",
             "dialogs_train": Path(options.replay_manifest).resolve(),
             "dialogs_validation": Path(options.adaptation_validation_manifest).resolve()}
    if any(p.is_relative_to(root) for p in paths.values()) or run.is_relative_to(root) or root == expansion:
        raise ValueError("Preparation output must be separate from inputs and training outputs")
    contract = dict(schema=SCHEMA, plan_sha256=plan_hash, padding_seconds=padding,
                    manifests={key: dict(path=str(path), sha256=file_digest(path)) for key,path in paths.items()},
                    recipe=json.loads(json.dumps(config)))
    root.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(root.parent / f".{root.name}.prepare.lock"), timeout=0):
        marker = root / "preparation_state.json"
        if root.exists() and any(root.iterdir()):
            if not marker.is_file():
                raise ValueError("Preparation output is not empty and has no matching state")
            state = json.loads(marker.read_text())
            if state.get("contract") != contract:
                raise ValueError("Preparation inputs/options changed; choose a new output directory")
        else:
            root.mkdir(parents=True, exist_ok=True)
            state = dict(contract=contract, alignment_parts={}, complete=False)
        state.update(status="running", complete=False, phase="audit-inputs")
        write_json(marker, state)
        try:
            log("[yo-retrain] Verifying existing WAV hashes and holdouts (CPU only).")
            groups = {key: checked_rows(path) for key,path in paths.items()}
            train, validation = groups["train"], groups["validation"]
            for name, rows in (("train", train), ("validation", validation)):
                if (len(rows) != imported[f"{name}_clips"] or not math.isclose(
                        sum(r["duration"] for r in rows)/3600, imported[f"{name}_hours"], abs_tol=1e-9)):
                    raise ValueError(f"YO {name} differs from the completed expansion report")
            original = train + validation
            if any(r.get("source_revision") != plan["revision"] for r in original):
                raise ValueError("YO WAV source revision differs from the local Parquet plan")
            wanted = {raw_id(r) for r in original}
            if len(wanted) != len(original):
                raise ValueError("Repeated YO utterance across train/validation")
            assert_disjoint(train, validation, "speaker")
            replay = groups["dialogs_train"]
            dialogs_root = paths["dialogs_train"].parent
            if paths["dialogs_validation"].parent != dialogs_root:
                raise ValueError("Reviewed Dialogs manifests must share an audit directory")
            audit_state = json.loads((dialogs_root / "review_state.json").read_text())
            audit_report = json.loads((dialogs_root / "audit_report.json").read_text())
            if (not audit_state.get("complete") or not audit_report.get("complete")
                    or any(audit_report.get("manifests", {}).get(split) != file_digest(paths[f"dialogs_{split}"])
                           for split in ("train", "validation"))):
                raise ValueError("Dialogs review is incomplete or its manifests have changed")
            if any(r.get("dialogs_tail_review", {}).get("method") != "dialogs-asr-vad-tail-v1"
                   for r in replay + groups["dialogs_validation"]):
                raise ValueError("Run dialogs_tail_review before preparing this fresh mixed training stage")
            holdout, collisions, without_reference = clean_dialogs_holdout(
                train, validation, replay, groups["dialogs_validation"], options.max_prompt_seconds)
            state.update(phase="alignment"); write_json(marker, state)
            alignments = collect_alignments(plan, wanted, root, state, log)
            needed = sum(Path(r["audio"]).stat().st_size for r in original)
            already_written = sum(p.stat().st_size for p in (root / "audio").glob("*.wav"))
            if shutil.disk_usage(root).free < max(0, needed-already_written) + 512*1024**2:
                raise OSError("Insufficient space for new WAV copies plus 512 MiB headroom")
            (root / "audio").mkdir(exist_ok=True)
            state.update(phase="trim"); write_json(marker, state)
            prepared = {}
            for name in ("train", "validation"):
                rows = []
                for index, row in enumerate(groups[name], 1):
                    record, shard = alignments[raw_id(row)]
                    rows.append(crop_row(row, record, shard, root, padding))
                    if index % 500 == 0 or index == len(groups[name]):
                        state.update(split=name, processed=index)
                        write_json(marker, state)
                        log(f"[yo-retrain] Trimmed/reused {name}: {index}/{len(groups[name])}")
                SpeakerPairs(rows, options.max_prompt_seconds, options.max_target_seconds)
                prepared[name] = rows
                write_manifest(root / f"{name}.jsonl", rows)
            write_manifest(root / "dialogs-train.jsonl", replay)
            write_manifest(root / "dialogs-validation.jsonl", holdout)
            config["manifest_path"] = str(root / "train.jsonl")
            config["output_dir"] = str(run)
            config["options"].update(validation_manifest=str(root / "validation.jsonl"),
                replay_manifest=str(root / "dialogs-train.jsonl"),
                adaptation_validation_manifest=str(root / "dialogs-validation.jsonl"))
            # Reload the exact files through the same route as precompute/train.
            datasets = prepare_run_data(config["manifest_path"], config["options"], config["seed"],
                                        include_adaptation_validation=True)
            primary = primary_examples_per_update(options, config["gradient_accumulation"])
            effective = options.batch_size * config["gradient_accumulation"]
            full, partial = divmod(len(train), primary)
            updates = math.ceil(len(train)/primary)
            removed = [r["tail_trim"]["removed_seconds"] for rows in prepared.values() for r in rows]
            report = dict(schema=SCHEMA, complete=True, padding_seconds=padding,
                initialization="fresh LoRA on original pretrained base; no prior adapter",
                datasets={name: dataset_summary(data.rows) for name,data in zip(
                    ("train", "validation", "dialogs_train", "dialogs_validation"), datasets)},
                before={name: dataset_summary(groups[name]) for name in ("train", "validation")},
                trimmed_clips=sum(s > 0 for s in removed), removed_hours=sum(removed)/3600,
                max_removed_seconds=max(removed), source_shards=len(plan["selected"]),
                dialogs_audio="ASR-matched timestamps plus acoustic VAD review",
                dialogs_review_actions=dict(Counter(
                    r['dialogs_tail_review']['action'] for r in replay + holdout)),
                dialogs_validation_excluded_text_matches=[r["utt_id"] for r in collisions],
                dialogs_validation_excluded_without_reference=without_reference,
                primary_examples_per_full_update=primary, replay_examples_per_full_update=effective-primary,
                replay_draws_per_epoch=full*(effective-primary)+round(partial*(effective-primary)/primary),
                updates_per_epoch=updates, total_steps=updates*config["training_epochs"],
                recipe=str(root / "recipe.json"))
            cropped = prepared["train"] + prepared["validation"]
            largest = sorted(cropped, key=lambda r: -r["tail_trim"]["removed_seconds"])[:10]
            sample = random.Random(config["seed"]).sample(cropped, min(20, len(cropped)))
            by_id = {r["utt_id"]: r for r in original}
            write_json(root / "tail_audit.json", [dict(utt_id=r["utt_id"], text=r["text"],
                original_audio=by_id[r["utt_id"]]["audio"], trimmed_audio=r["audio"], **r["tail_trim"])
                for r in {r["utt_id"]: r for r in largest + sample}.values()])
            write_json(root / "recipe.json", config)
            write_json(root / "report.json", report)
            state.update(status="completed", phase="complete", complete=True)
            write_json(marker, state)
            log(f"[yo-retrain] Complete: {len(train)} train, {len(validation)} validation; "
                f"removed {report['removed_hours']:.3f} hours. Recipe: {root / 'recipe.json'}")
            return report
        except BaseException as error:
            state.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                         complete=False, error_type=type(error).__name__, error=str(error))
            write_json(marker, state)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expansion", default="training/datasets/yo-half")
    parser.add_argument("--plan", default="training/plans/yo-half/plan.json")
    parser.add_argument("--output", default="training/datasets/yo-trimmed-06")
    parser.add_argument("--tail-padding-seconds", type=float, default=.2)
    parser.add_argument("--recipe", default="recipes/russian_trimmed_base_96gb.json")
    parser.add_argument("--review-dialogs", action="store_true", help="First run/resume ASR+VAD review of the original Dialogs selection")
    parser.add_argument("--asr-model", default="large-v3", help="Already cached Faster-Whisper model or local CTranslate2 directory")
    parser.add_argument("--asr-device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--asr-compute-type", default="int8")
    args = parser.parse_args()
    if args.review_dialogs:
        from .dialogs_tail_review import review_dialogs
        config = json.loads(Path(args.recipe).read_text())
        train = Path(config["options"]["replay_manifest"])
        validation = Path(config["options"]["adaptation_validation_manifest"])
        if train.name != "train.jsonl" or validation != train.parent / "validation.jsonl":
            parser.error("--review-dialogs requires train.jsonl and validation.jsonl in the same output directory")
        review_dialogs(output=train.parent, padding=args.tail_padding_seconds, model=args.asr_model,
                       device=args.asr_device, compute_type=args.asr_compute_type)
    print(json.dumps(prepare_retraining(args.expansion, args.plan, args.output,
        args.tail_padding_seconds, args.recipe), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

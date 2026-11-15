"""Prepare a fresh mixed YO/Dialogs run from existing tail-reviewed WAVs.

Filter complete recordings by their actual duration, preserve split membership,
and reuse audio/cache files. No downloads, additional crops or text limits.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import tempfile

from .training_data import SpeakerPairs, dataset_fingerprint, file_digest, write_manifest
from .training_run import TrainingOptions, prepare_run_data, primary_examples_per_update
from .yo_expansion import checked_rows, write_json
from .yo_retrain import SCHEMA as SOURCE_SCHEMA

SCHEMA = "duration-limited-retrain-v1"
MANIFESTS = {"train": "train.jsonl", "validation": "validation.jsonl",
             "dialogs_train": "dialogs-train.jsonl", "dialogs_validation": "dialogs-validation.jsonl"}


def summary(rows):
    return dict(clips=len(rows), hours=sum(r["duration"] for r in rows) / 3600,
                local_speakers=len({r["speaker_id"] for r in rows}),
                longest_clip_seconds=max((r["duration"] for r in rows), default=0.),
                fingerprint=dataset_fingerprint(rows))


def filter_duration(rows, max_seconds):
    if type(max_seconds) not in (int, float) or not math.isfinite(max_seconds) or max_seconds <= 0:
        raise ValueError("max_seconds must be positive and finite")
    candidates, excluded = [], []
    for row in rows:
        if not math.isfinite(row["duration"]) or row["duration"] <= 0:
            raise ValueError("Invalid actual audio duration")
        if row["duration"] > max_seconds:
            excluded.append({"utt_id": row["utt_id"], "reason": "duration_over_limit", "seconds": row["duration"]})
        else:
            candidates.append(row)
    groups = defaultdict(list)
    for row in candidates:
        groups[(row["speaker_id"], row["language"])].append(row)
    kept = []
    for row in candidates:
        # Both prompts and targets obey the same limit. Distinctness is symmetric,
        # so removing isolated rows cannot invalidate the other retained pairs.
        if any(SpeakerPairs.distinct(row, ref) for ref in groups[(row["speaker_id"], row["language"])]):
            kept.append(row)
        else:
            excluded.append({"utt_id": row["utt_id"], "reason": "no_distinct_reference_after_filter",
                             "seconds": row["duration"]})
    return kept, excluded


def prepare_stage(source="training/datasets/yo-trimmed-06", output="training/datasets/trimmed-20s-07",
                  max_seconds=20., recipe_path="recipes/russian_trimmed_20s_base_96gb.json", log=print):
    from filelock import FileLock
    # Validate even if the source is empty or unavailable.
    filter_duration([], max_seconds)
    source, output = Path(source).resolve(), Path(output).resolve()
    config = json.loads(Path(recipe_path).read_text(encoding="utf-8"))
    if config.get("initial_adapter") or config.get("resume_checkpoint"):
        raise ValueError("Fresh recipe must not contain an initial adapter or resume checkpoint")
    config["options"].update(max_prompt_seconds=max_seconds, max_target_seconds=max_seconds)
    options = TrainingOptions.parse(config["options"])
    if (config["length_mode"] != "Epochs" or options.sampling_strategy != "coverage_replay"
            or options.validation_scope != "speaker" or not options.expected_base_fingerprint
            or not options.adaptation_validation_manifest):
        raise ValueError("Require fresh pinned base, coverage_replay, Epochs and both holdouts")
    if (type(config["training_epochs"]) is not int or config["training_epochs"] < 1
            or type(config["gradient_accumulation"]) is not int or config["gradient_accumulation"] < 1):
        raise ValueError("Epochs and gradient accumulation must be positive integers")
    run = Path(config["output_dir"]).resolve()
    if (output.is_relative_to(source) or source.is_relative_to(output)
            or run.is_relative_to(output) or output.is_relative_to(run)
            or run.is_relative_to(source) or source.is_relative_to(run)):
        raise ValueError("Source, preparation output and training output must be separate")
    if run.exists() and any(run.iterdir()):
        raise ValueError("Training output is populated; choose a new run or resume its saved recipe")
    output.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output.parent / f".{output.name}.prepare.lock"), timeout=0):
        if output.exists():
            raise ValueError("Preparation output already exists; use its recipe or choose a new directory")
        paths = {name: source / filename for name, filename in MANIFESTS.items()}
        inputs = [source / "report.json", source / "preparation_state.json", *paths.values()]
        hashes = {str(p): file_digest(p) for p in inputs}
        source_report = json.loads((source / "report.json").read_text())
        source_state = json.loads((source / "preparation_state.json").read_text())
        if (source_report.get("schema") != SOURCE_SCHEMA or not source_report.get("complete")
                or not source_state.get("complete") or source_report.get("padding_seconds") != .2):
            raise ValueError("Finish the original final-word +200 ms preparation first")
        original = {}
        for name, path in paths.items():
            log(f"[duration-retrain] Checking {name} WAV hashes and source fingerprint...")
            rows = checked_rows(path)
            expected = source_report.get("datasets", {}).get(name, {})
            if not rows or dataset_fingerprint(rows) != expected.get("fingerprint"):
                raise ValueError(f"Source {name} differs from the completed preparation report")
            if name.startswith("dialogs_"):
                if any(r.get("dialogs_tail_review", {}).get("method") != "dialogs-asr-vad-tail-v1" for r in rows):
                    raise ValueError("Expected the previously reviewed Dialogs audio")
            elif any(r.get("tail_trim", {}).get("method") != SOURCE_SCHEMA
                     or r["tail_trim"].get("padding_seconds") != .2 for r in rows):
                raise ValueError("Expected the previously aligned and trimmed YO audio")
            original[name] = rows
        prepared, audit = {}, []
        for name, rows in original.items():
            selected, excluded = filter_duration(rows, max_seconds)
            if not selected:
                raise ValueError(f"No usable {name} pairs remain after the duration filter")
            SpeakerPairs(selected, max_seconds, max_seconds)
            prepared[name] = selected
            audit.extend(dict(split=name, **item) for item in excluded)
            log(f"[duration-retrain] {name}: {len(rows)} -> {len(selected)} clips, "
                f"{sum(r['duration'] for r in selected) / 3600:.2f} hours")
        with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
            staging = Path(temporary) / "stage"
            staging.mkdir()
            for name, filename in MANIFESTS.items():
                write_manifest(staging / filename, prepared[name])
            config["manifest_path"] = str(staging / MANIFESTS["train"])
            config["output_dir"] = str(run)
            config["options"].update(validation_manifest=str(staging / MANIFESTS["validation"]),
                replay_manifest=str(staging / MANIFESTS["dialogs_train"]),
                adaptation_validation_manifest=str(staging / MANIFESTS["dialogs_validation"]))
            # Exercise actual precompute/train checks: speakers, source overlap,
            # same-voice references and secondary-holdout transcript leakage.
            datasets = prepare_run_data(config["manifest_path"], config["options"], config["seed"],
                                        include_adaptation_validation=True)
            primary = primary_examples_per_update(options, config["gradient_accumulation"])
            effective = options.batch_size * config["gradient_accumulation"]
            full, partial = divmod(len(prepared["train"]), primary)
            updates = math.ceil(len(prepared["train"]) / primary)
            config["manifest_path"] = str(output / MANIFESTS["train"])
            config["options"].update(validation_manifest=str(output / MANIFESTS["validation"]),
                replay_manifest=str(output / MANIFESTS["dialogs_train"]),
                adaptation_validation_manifest=str(output / MANIFESTS["dialogs_validation"]))
            report = dict(schema=SCHEMA, complete=True, source=str(source), source_files=hashes,
                initialization="fresh LoRA on original pretrained base; no prior adapter",
                filters=dict(max_audio_seconds=max_seconds, max_text_characters=None,
                             duration_basis="actual WAV duration after existing tail review; no further cropping"),
                padding_seconds=.2, before={name: summary(rows) for name, rows in original.items()},
                datasets={name: summary(data.rows) for name, data in zip(MANIFESTS, datasets)},
                rejections={name: {reason: sum(a["split"] == name and a["reason"] == reason for a in audit)
                    for reason in ("duration_over_limit", "no_distinct_reference_after_filter")} for name in MANIFESTS},
                primary_examples_per_full_update=primary, replay_examples_per_full_update=effective-primary,
                replay_draws_per_epoch=full*(effective-primary)+round(partial*(effective-primary)/primary),
                updates_per_epoch=updates, training_epochs=config["training_epochs"],
                total_steps=updates*config["training_epochs"], recipe=str(output / "recipe.json"))
            if any(file_digest(path) != digest for path, digest in hashes.items()):
                raise ValueError("Source metadata changed during preparation")
            write_json(staging / "filter_audit.json", audit)
            write_json(staging / "report.json", report)
            write_json(staging / "recipe.json", config)
            if output.exists():
                raise ValueError("Preparation output was created by another process")
            staging.rename(output)
    log(f"[duration-retrain] Ready: {updates} updates/epoch; recipe {output / 'recipe.json'}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="training/datasets/yo-trimmed-06")
    parser.add_argument("--output", default="training/datasets/trimmed-20s-07")
    parser.add_argument("--max-seconds", type=float, default=20.)
    parser.add_argument("--recipe", default="recipes/russian_trimmed_20s_base_96gb.json")
    args = parser.parse_args()
    print(json.dumps(prepare_stage(args.source, args.output, args.max_seconds, args.recipe),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

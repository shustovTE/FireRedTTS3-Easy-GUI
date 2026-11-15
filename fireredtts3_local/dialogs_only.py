"""Fresh base adaptation on the original, untrimmed Dialogs selection."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path

from .dialogs_dataset import (SPEAKERS, checked_plan, lexical_key, prepare_dialogs,
                             summarize)
from .training_data import dataset_fingerprint, file_digest, write_manifest
from .training_run import TrainingOptions, prepare_run_data

SCHEMA = "dialogs-original-base-v1"
DEFAULT_OUTPUT = "training/datasets/dialogs-base-08"
DEFAULT_RECIPE = "recipes/russian_dialogs_base_96gb.json"


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def find_original_plan():
    previous = Path("training/datasets/dialogs-neutral/import_report.json")
    if not previous.is_file():
        raise ValueError("Original Dialogs import report missing; supply --plan with the existing Dialogs plan.json")
    digest = json.loads(previous.read_text())["plan_sha256"]
    candidates = sorted(Path("training/plans").glob("dialogs*/plan.json"))
    matches = [path for path in candidates if file_digest(path) == digest]
    if not matches:
        raise ValueError("Original Dialogs selection plan not found by hash; supply --plan")
    return matches[0]


def audit_prepared(output=DEFAULT_OUTPUT):
    """Check fingerprints and usable pairs before starting or deleting old data."""
    output = Path(output).resolve()
    report = json.loads((output / "report.json").read_text())
    config = json.loads((output / "recipe.json").read_text())
    if report.get("schema") != SCHEMA or not report.get("complete"):
        raise ValueError("Complete the original Dialogs-only preparation first")
    if file_digest(output / "recipe.json") != report["recipe_sha256"]:
        raise ValueError("Prepared recipe was changed; use its original recipe")
    train, validation, replay, adaptation = prepare_run_data(config["manifest_path"], config["options"],
        config["seed"], include_adaptation_validation=True)
    if replay is not None or adaptation is not None:
        raise ValueError("Dialogs-only run cannot depend on replay or a second corpus")
    for split, pairs in (("train", train), ("validation", validation)):
        if dataset_fingerprint(pairs.rows) != report["datasets"][split]["fingerprint"]:
            raise ValueError(f"Prepared {split} audio or manifest changed")
        if {row["speaker_id"] for row in pairs.rows} != {"dialogs:" + s for s in SPEAKERS}:
            raise ValueError("Require all three Dialogs actors and no other speakers")
    if {lexical_key(r["text"]) for r in train.rows} & {lexical_key(r["text"]) for r in validation.rows}:
        raise ValueError("Dialogs train/validation text leakage")
    return report, config, train.rows + validation.rows


def prepare(plan_path=None, output=DEFAULT_OUTPUT, recipe_path=DEFAULT_RECIPE, log=print):
    plan_path = Path(plan_path) if plan_path else find_original_plan()
    plan, source, _ = checked_plan(plan_path)
    policy = plan["report"]
    if (policy["max_seconds"] != 10. or policy["emotion"] != "neutral"
            or policy["train_minutes_per_speaker"] != 30.
            or not math.isclose(policy["validation_minutes_per_speaker"], 10/3)):
        raise ValueError("Require the existing neutral <=10s plan: 30min train + 3min20s validation per actor")
    output = Path(output).resolve()
    config = json.loads(Path(recipe_path).read_text())
    options = TrainingOptions.parse(config["options"])
    if (config.get("initial_adapter") or config.get("resume_checkpoint")
            or options.replay_manifest or options.adaptation_validation_manifest
            or options.sampling_strategy != "coverage" or options.validation_scope != "recording"
            or not options.validation_all_targets or not options.expected_base_fingerprint
            or options.max_prompt_seconds != 10. or options.max_target_seconds != 10.
            or config["length_mode"] != "Epochs"):
        raise ValueError("Require a fresh pinned-base recipe, Dialogs-only coverage and full recording holdout")
    run = Path(config["output_dir"]).resolve()
    if output.exists() or (run.exists() and any(run.iterdir())):
        raise ValueError("Preparation/training output must be new; existing runs are not overwritten")
    if (output.is_relative_to(source) or source.is_relative_to(output)
            or run.is_relative_to(output) or output.is_relative_to(run)):
        raise ValueError("Source audio, prepared data and training output must be separate")
    log(f"[dialogs-only] Reimporting whole original WAVs using {plan_path}; no tail review or trimming.")
    # The original importer preserves full recordings, resampling to 24 kHz.
    # Neither YO support manifests nor the reviewed Dialogs WAVs are supplied.
    prepare_dialogs(plan_path, output, log=log)
    config["manifest_path"] = str(output / "train.jsonl")
    config["output_dir"] = str(run)
    config["options"].update(validation_manifest=str(output / "validation.jsonl"),
                             cache_dir=str(Path(options.cache_dir).resolve()))
    train, validation, _, _ = prepare_run_data(config["manifest_path"], config["options"], config["seed"],
                                               include_adaptation_validation=True)
    datasets, dropped = {}, {}
    for split, pairs, minutes in (("train", train, 30.), ("validation", validation, 10/3)):
        rows = pairs.rows
        # Durations in the source CSV can differ slightly from decoded audio.
        # Enforce the real cap by removing complete triples, never cropping audio.
        removed = []
        for speaker in SPEAKERS:
            actor = "dialogs:" + speaker
            while sum(r["duration"] for r in rows if r["speaker_id"] == actor) > minutes*60 + 1e-6:
                triple = max(r["selection_triple"] for r in rows if r["speaker_id"] == actor)
                selected = [r for r in rows if r["speaker_id"] == actor and r["selection_triple"] == triple]
                removed.extend(r["utt_id"] for r in selected)
                rows = [r for r in rows if r not in selected]
            kept = [r for r in rows if r["speaker_id"] == actor]
            slots = Counter(r["selection_slot"] for r in kept)
            if not kept or set(slots) != {"question", "statement", "random"} or len(set(slots.values())) != 1:
                raise ValueError("An actor or exact count thirds were lost during import")
        if any(r.get("dialogs_tail_review") or r.get("tail_trim") or r["start"] != 0. for r in rows):
            raise ValueError("Expected original whole recordings")
        write_manifest(output / f"{split}.jsonl", rows)
        datasets[split] = dict(clips=len(rows), hours=sum(r["duration"] for r in rows)/3600,
            fingerprint=dataset_fingerprint(rows), actors={s: summarize([r for r in rows
                if r["speaker_id"] == "dialogs:"+s]) for s in SPEAKERS})
        dropped[split] = removed
    prepare_run_data(config["manifest_path"], config["options"], config["seed"])
    write_json(output / "recipe.json", config)
    updates = math.ceil(datasets["train"]["clips"] / (options.batch_size * config["gradient_accumulation"]))
    report = dict(schema=SCHEMA, complete=True, source_dir=str(source), plan=str(plan_path.resolve()),
        plan_sha256=file_digest(plan_path), source_revision=policy["revision"],
        initialization="original pretrained base; fresh LoRA; no previous adapter or YO replay",
        audio_processing="whole original WAV; resample 24kHz PCM16; no tail trim, VAD, gain change or normalization",
        validation="all held-out targets; known actors, disjoint original sessions and word sequences",
        datasets=datasets, duration_budget_removed_triples=dropped,
        updates_per_epoch=updates, maximum_steps=updates*config["training_epochs"],
        recipe=str(output / "recipe.json"), recipe_sha256=file_digest(output / "recipe.json"))
    write_json(output / "report.json", report)
    audit_prepared(output)
    log(f"[dialogs-only] Ready: {updates} updates/epoch, up to {report['maximum_steps']} steps.")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", help="Existing original Dialogs plan; default: locate by old import-report hash")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--recipe", default=DEFAULT_RECIPE)
    parser.add_argument("--audit", action="store_true", help="Check an already prepared run")
    args = parser.parse_args()
    report = audit_prepared(args.output)[0] if args.audit else prepare(args.plan, args.output, args.recipe)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

"""Prepare the expanded YO stage, retaining Dialogs replay and a clean holdout."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import tempfile

from .dialogs_dataset import lexical_key
from .training_data import SpeakerPairs, assert_disjoint, dataset_fingerprint, write_manifest
from .training_run import (TrainingOptions, initial_adapter_info, prepare_run_data,
                           primary_examples_per_update)
from .yo_expansion import checked_rows, write_json


def clean_dialogs_holdout(train, validation, replay, original_holdout, max_prompt_seconds=None):
    """Keep actor/session holdout independent of both training corpora."""
    assert_disjoint(train + replay, original_holdout, "recording")
    assert_disjoint(validation, original_holdout, "speaker")
    training_texts = {lexical_key(r["text"]) for r in train + replay}
    collisions = [r for r in original_holdout if lexical_key(r["text"]) in training_texts]
    candidates = [r for r in original_holdout if lexical_key(r["text"]) not in training_texts]
    groups = defaultdict(list)
    for row in candidates:
        groups[(row["speaker_id"], row["language"])].append(row)
    holdout = [r for group in groups.values() for r in group
               if any(SpeakerPairs.distinct(r, ref) for ref in group
                      if max_prompt_seconds is None or ref["duration"] <= max_prompt_seconds)]
    if not holdout or {r["speaker_id"] for r in holdout} != {r["speaker_id"] for r in original_holdout}:
        raise ValueError("Too few independent Dialogs holdout pairs; prepare additional validation before training")
    return holdout, collisions, len(candidates) - len(holdout)


def prepare_stage(expansion, output, recipe_path="recipes/russian_yo_half_96gb.json"):
    """Audit on CPU, keep every YO target, filter only the secondary holdout.

    A new YO transcript can repeat a held-out Dialogs sentence. Exclude that
    sentence from this stage's metric rather than silently reducing YO coverage.
    Original manifests/WAVs are never rewritten. The final recipe and reports
    appear together only after all checks pass.
    """
    expansion, output = Path(expansion).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError("Stage output must be a new directory")
    state = json.loads((expansion / "expansion_state.json").read_text())
    imported = json.loads((expansion / "split_report.json").read_text())
    if not state.get("complete") or state.get("plan_sha256") != imported.get("plan_sha256"):
        raise ValueError("Finish/resume the YO expansion before preparing training")

    config = json.loads(Path(recipe_path).read_text())
    config["manifest_path"] = str(expansion / "train.jsonl")
    config["options"]["validation_manifest"] = str(expansion / "validation.jsonl")
    options = TrainingOptions.parse(config["options"])
    if options.sampling_strategy != "coverage_replay" or config["length_mode"] != "Epochs":
        raise ValueError("The expanded YO stage requires coverage_replay in Epochs mode")
    train = checked_rows(config["manifest_path"])
    validation = checked_rows(options.validation_manifest)
    for name, rows in (("train", train), ("validation", validation)):
        hours = sum(r["duration"] for r in rows) / 3600
        if (len(rows) != imported[f"{name}_clips"]
                or not math.isclose(hours, imported[f"{name}_hours"], rel_tol=1e-9, abs_tol=1e-9)):
            raise ValueError(f"YO {name} no longer matches split_report.json")
    replay = checked_rows(options.replay_manifest)
    original_holdout = checked_rows(options.adaptation_validation_manifest)
    if not train or not validation or not replay or not original_holdout:
        raise ValueError("YO train/validation and Dialogs train/validation must be nonempty")
    holdout, collisions, without_reference = clean_dialogs_holdout(
        train, validation, replay, original_holdout, options.max_prompt_seconds)
    initial = initial_adapter_info(config["initial_adapter"], config["rank"], config["alpha"],
                                   config["dropout"], options.profile)
    config["initial_adapter"] = initial["path"]
    config["options"]["replay_manifest"] = str(Path(options.replay_manifest).resolve())

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".yo-stage-", dir=output.parent) as temporary:
        staging = Path(temporary) / "stage"
        staging.mkdir()
        write_manifest(staging / "dialogs-validation.jsonl", holdout)
        config["options"]["adaptation_validation_manifest"] = str(staging / "dialogs-validation.jsonl")
        # Exercise the actual training/precompute data path before emitting a recipe.
        datasets = prepare_run_data(config["manifest_path"], config["options"], config["seed"],
                                    include_adaptation_validation=True)
        primary = primary_examples_per_update(options, config["gradient_accumulation"])
        effective = options.batch_size * config["gradient_accumulation"]
        full, partial = divmod(len(train), primary)
        updates = math.ceil(len(train) / primary)
        report = {
            "expansion_plan_sha256": imported["plan_sha256"],
            "initial_adapter": initial,
            "datasets": {
                name: {"clips": len(data.rows), "hours": sum(r["duration"] for r in data.rows) / 3600,
                       "local_speakers": len({r["speaker_id"] for r in data.rows}),
                       "longest_clip_seconds": max(r["duration"] for r in data.rows),
                       "fingerprint": dataset_fingerprint(data.rows)}
                for name, data in zip(("train", "validation", "replay", "dialogs_validation"), datasets)
            },
            "dialogs_validation_excluded_text_matches": [r["utt_id"] for r in collisions],
            "dialogs_validation_excluded_without_reference": without_reference,
            "primary_examples_per_full_update": primary,
            "replay_examples_per_full_update": effective - primary,
            "replay_draws_per_epoch": full * (effective - primary) + round(partial * (effective - primary) / primary),
            "updates_per_epoch": updates,
            "training_epochs": config["training_epochs"],
            "total_steps": updates * config["training_epochs"],
            "yo_validation_pairs": options.validation_pairs,
            "recipe": str(output / "recipe.json"),
        }
        config["options"]["adaptation_validation_manifest"] = str(output / "dialogs-validation.jsonl")
        write_json(staging / "recipe.json", config)
        write_json(staging / "report.json", report)
        if output.exists():
            raise ValueError("Stage output was created by another process")
        staging.rename(output)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expansion", default="training/datasets/yo-half")
    parser.add_argument("--output", default="training/plans/yo-half-training")
    parser.add_argument("--recipe", default="recipes/russian_yo_half_96gb.json")
    args = parser.parse_args()
    print(json.dumps(prepare_stage(args.expansion, args.output, args.recipe), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

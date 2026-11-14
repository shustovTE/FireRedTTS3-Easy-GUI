"""Reproducible preparation/training commands: python -m fireredtts3_local.adapt."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from .training_data import SpeakerPairs, load_manifest, split_manifest, write_manifest
from .training_run import TrainingOptions, prepare_run_data, train_lora


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare-yo", help="Import aligned local YO-CPT-ru shards")
    prepare.add_argument("inputs", nargs="+")
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--source-revision", required=True)
    prepare.add_argument("--min-mos", type=float, default=4.0)
    prepare.add_argument("--max-seconds", type=float, default=10.0)
    prepare.add_argument("--max-hours", type=float, default=50.0)
    prepare.add_argument("--max-speaker-minutes", type=float, default=30.0)
    split = commands.add_parser("split", help="Create disjoint train and validation manifests")
    split.add_argument("manifest")
    split.add_argument("--output", required=True)
    split.add_argument("--fraction", type=float, default=.1)
    split.add_argument("--scope", choices=["speaker", "recording"], default="speaker")
    split.add_argument("--seed", type=int, default=1234)
    audit = commands.add_parser("audit", help="Validate data and pairing without GPU/model downloads")
    audit.add_argument("manifest")
    audit.add_argument("--options")
    audit.add_argument("--seed", type=int, default=1234)
    train = commands.add_parser("train", help="Train from a JSON recipe")
    train.add_argument("recipe", help="JSON containing train_lora keyword arguments, including options")
    defaults = commands.add_parser("options", help="Print all adaptation option defaults")
    evaluate = commands.add_parser("evaluate", help="Generate an unseen-voice benchmark and optional metrics")
    evaluate.add_argument("benchmark")
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--adapter")
    evaluate.add_argument("--speaker-model", help="e.g. microsoft/wavlm-base-plus-sv (independent of CAM++)")
    evaluate.add_argument("--asr-model", help="Faster-Whisper model or local path")
    compare = commands.add_parser("compare", help="Compare fixed baseline and adapted benchmark reports")
    compare.add_argument("baseline")
    compare.add_argument("adapted")
    compare.add_argument("--max-sim-drop", type=float, default=.02)
    compare.add_argument("--max-wer-increase", type=float, default=.01)
    export = commands.add_parser("export", help="Merge the entire Base core and package for SGLang Omni")
    export.add_argument("--base", required=True, help="Original FireRedTeam/FireRedTTS3 directory")
    export.add_argument("--adapter", required=True)
    export.add_argument("--sglang-config", required=True, help="Compatible CKermit config.json")
    export.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "prepare-yo":
        from .yo_dataset import import_yo
        import_yo(args.inputs, args.output, args.min_mos, args.max_seconds,
                  args.max_hours, args.max_speaker_minutes, args.source_revision)
    elif args.command == "split":
        root = Path(args.output)
        if root.exists() and any(root.iterdir()):
            parser.error("split output must be empty")
        train_rows, validation = split_manifest(load_manifest(args.manifest), args.fraction, args.seed, args.scope)
        SpeakerPairs(train_rows); SpeakerPairs(validation)
        write_manifest(root / "train.jsonl", train_rows)
        write_manifest(root / "validation.jsonl", validation)
        print(json.dumps({"train_clips": len(train_rows), "validation_clips": len(validation), "scope": args.scope}))
    elif args.command == "audit":
        options = TrainingOptions.parse(Path(args.options).read_text() if args.options else None)
        train_pairs, val_pairs, replay_pairs = prepare_run_data(args.manifest, options, args.seed)
        print(json.dumps({"train_clips": len(train_pairs.rows), "train_speakers": len(train_pairs.groups),
                          "validation_clips": len(val_pairs.rows), "validation_speakers": len(val_pairs.groups),
                          "validation_scope": options.validation_scope,
                          "replay_clips": len(replay_pairs.rows) if replay_pairs else 0}, indent=2))
    elif args.command == "options":
        print(json.dumps(asdict(TrainingOptions()), indent=2))
    elif args.command == "train":
        from .model_manager import ModelManager
        config = json.loads(Path(args.recipe).read_text(encoding="utf-8"))
        train_lora(ModelManager(), **config)
    elif args.command == "evaluate":
        from .evaluation import evaluate_benchmark
        evaluate_benchmark(args.benchmark, args.output, args.adapter, args.speaker_model, args.asr_model)
    elif args.command == "compare":
        from .evaluation import compare_reports
        report = compare_reports(args.baseline, args.adapted, args.max_sim_drop, args.max_wer_increase)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if not report["automatic_gate_passed"]:
            raise SystemExit(1)
    elif args.command == "export":
        from .export import export_merged
        export_merged(args.base, args.adapter, args.sglang_config, args.output)


if __name__ == "__main__":
    main()

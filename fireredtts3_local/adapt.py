"""Reproducible preparation/training commands: python -m fireredtts3_local.adapt."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from .training_data import (SpeakerPairs, dataset_fingerprint, load_manifest,
                            split_manifest, split_with_anchors, write_manifest)
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
    prepare.add_argument("--segmentation", choices=["bounded", "complete-utterance"], default="bounded",
                         help="complete-utterance: whole records ending in . ! ?, no ellipsis or duration cap")
    prepare.add_argument("--tail-padding-seconds", type=float,
                         help="Complete utterances: end after final aligned word + this margin (e.g. 0.2)")
    prepare.add_argument("--max-hours", type=float, default=50.0, help="Total accepted hours; 0 processes all supplied shards")
    prepare.add_argument("--max-speaker-minutes", type=float, default=30.0)
    dialogs = commands.add_parser('plan-dialogs', help='Plan neutral <=10s Dialogs thirds with extra session-disjoint validation')
    dialogs.add_argument('--source', default='training/raw/dialogs')
    dialogs.add_argument('--output', default='training/plans/dialogs-neutral')
    dialogs.add_argument('--source-revision', default='a74ff44d8d27ca3d21098ea3ed4d31a32923fbdd')
    dialogs.add_argument('--max-seconds', type=float, default=10.)
    dialogs.add_argument('--train-minutes', type=float, default=30.)
    dialogs.add_argument('--validation-minutes', type=float, default=10/3)
    dialogs.add_argument('--seed', type=int, default=1234)
    dialogs.add_argument('--download', action='store_true', help='Fetch pinned CSV metadata, license and full file inventory; no audio')
    dialogs_download = commands.add_parser('download-dialogs', help='Resume a fixed plan with serial HTTP, verified cache reuse and rate backoff')
    dialogs_download.add_argument('plan')
    dialogs_download.add_argument('--interval', type=float, default=2., help='Minimum seconds between new download requests')
    dialogs_download.add_argument('--max-retries', type=int, default=8, help='Retries per file after temporary network/rate errors')
    dialogs_import = commands.add_parser('prepare-dialogs', help='Download/audit planned WAVs, keeping exact thirds')
    dialogs_import.add_argument('plan')
    dialogs_import.add_argument('--output', default='training/datasets/dialogs-neutral')
    dialogs_import.add_argument('--download', action='store_true', help='Download only planned audio files at the pinned revision')
    dialogs_import.add_argument('--replay-manifest', help='Existing YO train manifest; select complete <=10s replay with valid pairs')
    dialogs_import.add_argument('--validation-manifest', help='Existing unseen-speaker YO validation; select complete <=10s paired clips')
    split = commands.add_parser("split", help="Create disjoint train and validation manifests")
    split.add_argument("manifest")
    split.add_argument("--output", required=True)
    split.add_argument("--fraction", type=float, default=.1)
    split.add_argument("--scope", choices=["speaker", "recording"], default="speaker")
    split.add_argument("--seed", type=int, default=1234)
    split.add_argument("--unlimited-duration", action="store_true", help="Allow whole targets and references of any duration")
    split.add_argument("--anchor-train", help="Previous train manifest: preserve known identities in train")
    split.add_argument("--anchor-validation", help="Previous validation manifest: preserve its held-out identities")
    audit = commands.add_parser("audit", help="Validate data and pairing without GPU/model downloads")
    audit.add_argument("manifest")
    audit.add_argument("--options")
    audit.add_argument("--seed", type=int, default=1234)
    train = commands.add_parser("train", help="Train from a JSON recipe")
    train.add_argument("recipe", help="JSON containing train_lora keyword arguments, including options")
    train.add_argument("--stop-after", type=int, help="Stop after N completed updates in this invocation; resume with the same recipe")
    precompute = commands.add_parser("precompute", help="Precompute frozen RedAE, CAM++ and PatchEncoder features")
    precompute.add_argument("recipe", help="Use the exact recipe/split intended for training")
    precompute.add_argument("--base", help="Local original weights; defaults to models/FireRedTTS3")
    precompute.add_argument("--repair", action="store_true", help="Recompute invalid entries; valid entries are reused")
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
    export = commands.add_parser("export", help="Merge the full core and build a verified SGLang Omni release")
    export.add_argument("--base", required=True, help="Original FireRedTeam/FireRedTTS3 directory")
    export.add_argument("--adapter", required=True)
    export.add_argument("--sglang-config", help="Legacy local-config export, without the verified Hub release workflow")
    export.add_argument("--output", required=True)
    export.add_argument("--sglang-repo", default="CKermit/FireRedTTS3-Base")
    export.add_argument("--sglang-revision", default="53f8b59c3252f15d539a9205fbcab3e334a36b9f")
    export.add_argument("--template-cache", default=".runtime/export-templates")
    export.add_argument("--evaluation-baseline", help="Optional existing baseline report, summarized without reference audio/text")
    export.add_argument("--evaluation-adapted", help="Existing report for the adapter being merged")
    publish = commands.add_parser("publish", help="Upload a verified model release to the account of the local HF token")
    publish.add_argument("export_dir")
    publish.add_argument("--repo-name", default="FireRedTTS3-Base-Russian")
    publish.add_argument("--public", action="store_true", help="Create a public model repository (otherwise private)")
    args = parser.parse_args()
    if args.command == "prepare-yo":
        from .yo_dataset import import_yo
        import_yo(args.inputs, args.output, args.min_mos, args.max_seconds,
                  args.max_hours, args.max_speaker_minutes, args.source_revision,
                  segmentation=args.segmentation, tail_padding_seconds=args.tail_padding_seconds)
    elif args.command == 'plan-dialogs':
        from .dialogs_dataset import plan_dialogs
        report = plan_dialogs(args.source, args.output, args.source_revision, args.max_seconds,
                              args.train_minutes, args.validation_minutes, args.seed, args.download)
        print(json.dumps(report, ensure_ascii=False, indent=2))
    elif args.command == 'download-dialogs':
        from .dialogs_dataset import download_dialogs
        print(json.dumps(download_dialogs(args.plan, args.interval, args.max_retries), ensure_ascii=False, indent=2))
    elif args.command == 'prepare-dialogs':
        from .dialogs_dataset import prepare_dialogs
        print(json.dumps(prepare_dialogs(args.plan, args.output, args.download,
              replay_manifest=args.replay_manifest, validation_manifest=args.validation_manifest), ensure_ascii=False, indent=2))
    elif args.command == "split":
        root = Path(args.output)
        if root.exists() and any(root.iterdir()):
            parser.error("split output must be empty")
        rows = load_manifest(args.manifest)
        anchors = {}
        if bool(args.anchor_train) != bool(args.anchor_validation):
            parser.error("Supply both --anchor-train and --anchor-validation")
        if args.anchor_train:
            old_train, old_val = load_manifest(args.anchor_train), load_manifest(args.anchor_validation)
            train_rows, validation = split_with_anchors(rows, old_train, old_val, args.fraction, args.seed, args.scope)
            anchors = {"train_fingerprint": dataset_fingerprint(old_train),
                       "validation_fingerprint": dataset_fingerprint(old_val)}
        else:
            train_rows, validation = split_manifest(rows, args.fraction, args.seed, args.scope)
        limits = (None, None) if args.unlimited_duration else (12.0, 20.0)
        SpeakerPairs(train_rows, *limits); SpeakerPairs(validation, *limits)
        write_manifest(root / "train.jsonl", train_rows)
        write_manifest(root / "validation.jsonl", validation)
        report = {"train_clips": len(train_rows), "validation_clips": len(validation), "scope": args.scope,
                  "train_hours": sum(r['duration'] for r in train_rows)/3600,
                  "validation_hours": sum(r['duration'] for r in validation)/3600,
                  "longest_clip_seconds": max(r['duration'] for r in rows), "anchors": anchors}
        (root / "split_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report))
    elif args.command == "audit":
        options = TrainingOptions.parse(Path(args.options).read_text() if args.options else None)
        train_pairs, val_pairs, replay_pairs, adaptation = prepare_run_data(args.manifest, options, args.seed, include_adaptation_validation=True)
        print(json.dumps({"train_clips": len(train_pairs.rows), "train_speakers": len(train_pairs.groups),
                          "validation_clips": len(val_pairs.rows), "validation_speakers": len(val_pairs.groups),
                          "validation_scope": options.validation_scope,
                          "replay_clips": len(replay_pairs.rows) if replay_pairs else 0,
                          "adaptation_validation_clips": len(adaptation.rows) if adaptation else 0}, indent=2))
    elif args.command == "options":
        print(json.dumps(asdict(TrainingOptions()), indent=2))
    elif args.command == "train":
        import signal
        from .model_manager import ModelManager
        from .training import STOP_EVENT
        config = json.loads(Path(args.recipe).read_text(encoding="utf-8"))
        if args.stop_after is not None:
            if args.stop_after < 1:
                parser.error("--stop-after must be positive")
            completed = 0
            def stop_after(*_):
                nonlocal completed
                completed += 1
                if completed >= args.stop_after:
                    STOP_EVENT.set()
            config['progress_cb'] = stop_after
        def stop_signal(signum, _frame):
            print(f"[training] Signal {signum}: finish the current update and save a resumable checkpoint.", flush=True)
            STOP_EVENT.set()
        previous = {sig: signal.signal(sig, stop_signal) for sig in (signal.SIGTERM, signal.SIGINT)}
        try:
            train_lora(ModelManager(), **config)
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
    elif args.command == "precompute":
        from .paths import MODEL_DIR
        from .training_features import precompute_recipe
        precompute_recipe(args.recipe, args.base or MODEL_DIR, repair=args.repair)
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
        if args.sglang_config:
            if args.evaluation_baseline or args.evaluation_adapted:
                parser.error("Evaluation summaries require the verified Hub-template export (omit --sglang-config)")
            from .export import export_merged
            export_merged(args.base, args.adapter, args.sglang_config, args.output)
        else:
            from .release import build_release
            print(json.dumps(build_release(args.base, args.adapter, args.output,
                template_repo=args.sglang_repo, template_revision=args.sglang_revision,
                template_cache=args.template_cache, evaluation_baseline=args.evaluation_baseline,
                evaluation_adapted=args.evaluation_adapted), ensure_ascii=False, indent=2))
    elif args.command == "publish":
        from .release import publish_release
        print(json.dumps(publish_release(args.export_dir, args.repo_name, public=args.public), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

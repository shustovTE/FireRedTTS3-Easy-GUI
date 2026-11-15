"""Remove known local YO datasets/caches after verifying the fresh Dialogs run.

Default is an inventory. --apply deletes the printed targets without a prompt.
Models, adapters, exported releases, logs and generated evaluation audio remain.
"""
from __future__ import annotations

import argparse
import contextlib
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil

from .dialogs_only import DEFAULT_OUTPUT, audit_prepared

PROJECT_TARGETS = (
    "training/raw/yo", "training/raw/yo-half",
    "training/datasets/yo", "training/datasets/yo-split",
    "training/datasets/yo-complete", "training/datasets/yo-complete-split",
    "training/datasets/yo-half", "training/datasets/yo-trimmed-06",
    "training/datasets/trimmed-20s-07",
    "training/plans/yo-half", "training/plans/yo-half-training",
    "training/datasets/dialogs-neutral/replay.jsonl",
    "training/datasets/dialogs-neutral/unseen-validation.jsonl",
    ".runtime/training-cache",
)
HUB_REPO = "datasets--NCSpeech--YO-CPT-ru"


def hub_roots():
    from huggingface_hub.constants import HF_HUB_CACHE
    return {Path(HF_HUB_CACHE), Path.home()/".cache/huggingface/hub",
            Path(os.environ.get("HF_CACHE_DIR", "/opt/hf-cache"))/"hub"}


def candidates(project, hubs):
    project = Path(project).resolve()
    paths = []
    for name in PROJECT_TARGETS:
        path = project / name
        if path.is_symlink() or (path.exists() and not path.resolve().is_relative_to(project)):
            raise ValueError(f"Refusing a linked/external project deletion target: {path}")
        if path.exists():
            paths.append(path)
    for hub in hubs:
        path = Path(hub).expanduser().resolve() / HUB_REPO
        if path.is_symlink():
            raise ValueError(f"Refusing a linked Hub repository: {path}")
        if path.exists():
            paths.append(path)
    return sorted(set(paths))


def check_protected(targets, protected):
    for target in targets:
        target = target.resolve()
        for item in protected:
            path = Path(item).resolve()
            if path == target or path.is_relative_to(target) or target.is_relative_to(path):
                raise ValueError(f"Deletion overlaps protected Dialogs/model/run files: {target} / {path}")


def inventory(path):
    if path.is_file():
        return dict(path=str(path), files=1, bytes=path.stat().st_size)
    count, size = 0, 0
    for parent, _, files in os.walk(path, followlinks=False):
        for name in files:
            item = Path(parent) / name
            count += 1
            size += item.lstat().st_size
    return dict(path=str(path), files=count, bytes=size)


def assert_no_direct_workers():
    """Launcher locks do not cover foreground imports/train/precompute commands."""
    blocked = {"fireredtts3_local.yo_expansion", "fireredtts3_local.yo_retrain",
               "fireredtts3_local.yo_training", "fireredtts3_local.duration_retrain"}
    for proc in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            args = proc.read_bytes().decode(errors="replace").split("\0")
        except (OSError, ProcessLookupError):
            continue
        if "-m" not in args:
            continue
        index = args.index("-m") + 1
        module = args[index] if index < len(args) else ""
        command = args[index+1] if index+1 < len(args) else ""
        if module in blocked or (module == "fireredtts3_local.adapt" and
                command in {"train", "precompute", "prepare-yo"}):
            raise ValueError(f"Data/precompute/training process is active (PID {proc.parent.name}); finish it before cleanup")


def cleanup(project=".", prepared=DEFAULT_OUTPUT, *, apply=False, hubs=None):
    import fcntl
    project = Path(project).resolve()
    if not (project/"fireredtts3_local").is_dir():
        raise ValueError("Run from the FireRed tuning repository")
    prepared = Path(prepared).resolve()
    report, config, rows = audit_prepared(prepared)
    targets = candidates(project, hub_roots() if hubs is None else hubs)
    protected = [project/"models", project/"training/outputs", project/"training/exports",
                 project/"training/jobs", project/"training/eval", prepared,
                 report["source_dir"], config["options"]["cache_dir"],
                 *[row["audio"] for row in rows]]
    check_protected(targets, protected)
    with contextlib.ExitStack() as stack:
        # Keep existing launcher and YO importer locks until deletion finishes.
        locks = list((project/".runtime/training-job-locks").glob("*.lock"))
        locks += [path/".yo-expansion.lock" for path in targets
                  if (path/".yo-expansion.lock").is_file()]
        for path in locks:
            stream = stack.enter_context(path.open("r+"))
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise ValueError(f"An active job holds {path}; no data deleted") from exc
        if apply:
            assert_no_direct_workers()
        entries = [inventory(path) for path in targets]
        result = dict(apply=apply, targets=entries, files=sum(r["files"] for r in entries),
                      bytes=sum(r["bytes"] for r in entries),
                      gib=round(sum(r["bytes"] for r in entries)/1024**3, 2),
                      note="Known YO raw/processed data and shared training feature cache only. "
                           "Models, historical adapters/exports/logs/evaluation outputs and Dialogs remain. "
                           "Old YO runs cannot resume after deletion. Custom locations/shared Xet chunks are not scanned.")
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        if apply:
            # Save the exact deletion inventory before mutating data.
            receipts = project/"training/cleanup"
            receipts.mkdir(parents=True, exist_ok=True)
            receipt = receipts/("yo-"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")+".json")
            result.update(deleted=[], complete=False)
            receipt.write_text(json.dumps(result, indent=2)+"\n")
            for path in targets:
                # Recheck immediately before deletion; never follow a replaced root.
                if path.is_symlink():
                    raise ValueError(f"Deletion target became a symlink: {path}")
                check_protected([path], protected)
                if path.is_dir():
                    shutil.rmtree(path)
                elif path.exists():
                    path.unlink()
                result["deleted"].append(str(path))
                receipt.write_text(json.dumps(result, indent=2)+"\n")
            audit_prepared(prepared)
            result["complete"] = True
            receipt.write_text(json.dumps(result, indent=2)+"\n")
            print(f"[cleanup-yo] Complete. Receipt: {receipt}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", default=DEFAULT_OUTPUT)
    parser.add_argument("--apply", action="store_true", help="Delete the inventoried YO data and old shared feature cache")
    args = parser.parse_args()
    cleanup(prepared=args.prepared, apply=args.apply)


if __name__ == "__main__":
    main()

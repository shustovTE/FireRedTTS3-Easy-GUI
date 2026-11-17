#!/usr/bin/env python3
"""Restore old absolute paths after moving the whole checkout, without editing data.

Run with system Python: the moved virtual environments may themselves need this
compatibility link. This does not move files, change fingerprints or replace an
existing directory/link.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def restore_alias(old_root, new_root):
    old = Path(os.path.abspath(os.path.expanduser(str(old_root))))
    new = Path(new_root).resolve(strict=True)
    if not (new / "fireredtts3_local").is_dir() or not (new / "README.md").is_file():
        raise ValueError("Destination is not a FireRedTTS3-Easy-GUI checkout")
    if old == new:
        return {"status": "already_at_destination", "root": str(new)}
    if os.path.lexists(old):
        if old.is_symlink() and old.resolve() == new:
            return {"status": "already_linked", "old_root": str(old), "root": str(new)}
        raise ValueError(f"Refusing to replace existing path: {old}. No files were changed.")
    if not old.parent.is_dir():
        raise ValueError(f"Old parent directory does not exist: {old.parent}")
    if old.parent.resolve().is_relative_to(new):
        raise ValueError("The compatibility link must be outside the current checkout")
    # symlink() is exclusive: a concurrent creator cannot be overwritten.
    old.symlink_to(new, target_is_directory=True)
    return {"status": "linked", "old_root": str(old), "root": str(new),
            "note": "Keep this link while historical manifests/venvs refer to the old root. "
                    "No audio, manifests, recipes or checkpoints were rewritten."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-root", default="/home/firered-tuning")
    args = parser.parse_args()
    try:
        result = restore_alias(args.old_root, Path(__file__).resolve().parents[1])
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Relocation compatibility failed: {exc}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

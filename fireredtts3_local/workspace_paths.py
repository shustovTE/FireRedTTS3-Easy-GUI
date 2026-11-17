"""Read historical workspace paths without a filesystem compatibility symlink.

Saved manifests/reports remain immutable. New runtime artifacts use the checkout
that contains this module; only the known previous workspace prefix is mapped.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
LEGACY_ROOT = Path("/home/firered-tuning")


def local_path(value):
    """Rebase the old absolute prefix, preserving venv interpreter symlinks."""
    path = Path(value).expanduser()
    if path.is_relative_to(LEGACY_ROOT):
        suffix = path.relative_to(LEGACY_ROOT)
        if ".." in suffix.parts:
            raise ValueError("Historical path must not escape the workspace")
        return WORKSPACE_ROOT / suffix
    return path


def recorded_path(value, namespace):
    """Recover the original path spelling for immutable provenance comparison."""
    actual = local_path(value).resolve()
    if (Path(namespace).is_relative_to(LEGACY_ROOT)
            and actual.is_relative_to(WORKSPACE_ROOT.resolve())):
        return LEGACY_ROOT / actual.relative_to(WORKSPACE_ROOT.resolve())
    return actual


def runtime_manifest(value):
    """Produce a content-addressed path view for the unmodified official loader.

    Text, order, durations and reference pairing are preserved. The caller must
    audit original data first and bind the returned file's hash to its recipe.
    """
    source = local_path(value).resolve(strict=True)
    # A balanced epoch can repeat a WAV thousands of times. Resolve every unique
    # path once and stream the JSONL, including when relocation needs a new copy.
    paths = {}
    def mapped_rows():
        with source.open(encoding="utf-8-sig") as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("Expected a JSON object in VoxCPM manifest")
                for key in ("audio", "ref_audio"):
                    name = row.get(key)
                    if not isinstance(name, str) or not name:
                        raise ValueError(f"VoxCPM manifest needs {key}")
                    if name not in paths:
                        path = local_path(name)
                        paths[name] = str((path if path.is_absolute() else source.parent / path).resolve(strict=True))
                    row[key] = paths[name]
                yield row

    for _ in mapped_rows():
        pass
    if not paths:
        raise ValueError("Empty VoxCPM manifest")
    if all(name == actual for name, actual in paths.items()):
        return source
    folder = WORKSPACE_ROOT / ".runtime/voxcpm-manifests"
    folder.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=folder, suffix=".tmp", delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        checksum = hashlib.sha256()
        with temporary_path.open("wb") as stream:
            for row in mapped_rows():
                line = (json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n").encode()
                checksum.update(line)
                stream.write(line)
        destination = folder / (checksum.hexdigest() + ".jsonl")
        try:
            os.link(temporary_path, destination)
        except FileExistsError:
            from .training_data import file_digest
            if destination.is_symlink() or file_digest(destination) != checksum.hexdigest():
                raise ValueError("Runtime manifest changed; refusing to replace it") from None
    finally:
        temporary_path.unlink(missing_ok=True)
    return destination

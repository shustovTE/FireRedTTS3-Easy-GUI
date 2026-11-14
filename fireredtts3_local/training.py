from __future__ import annotations

import json
import shutil
import threading
import warnings
from pathlib import Path



warnings.filterwarnings(
    "ignore",
    message=r"In 2\.9, this function's implementation will be changed to use .*torchaudio\.load_with_torchcodec.*",
    category=UserWarning,
)

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus"}
STOP_EVENT = threading.Event()


def _safe_name(name: str) -> str:
    out = "".join(c if c.isalnum() or c in "-_" else "_" for c in (name or "").strip())
    return out.strip("_") or "project"


def scan_dataset(source_dir: str):
    src = Path(source_dir).expanduser()
    if not src.exists():
        raise ValueError("Dataset source folder does not exist.")
    files = sorted(p for p in src.rglob("*") if p.is_file() and p.suffix.lower() in AUDIO_EXTS)
    with_text = sum(1 for p in files if p.with_suffix(".txt").exists())
    return files, with_text


def prepare_dataset(
    source_dir: str,
    project_name: str,
    projects_root: Path,
    language: str,
    asr_manager=None,
    asr_model: str | None = None,
    asr_language: str = "Auto-detect",
    asr_batch_size: int = 8,
    transcribe_missing: bool = True,
    log=print,
    speaker_id: str | None = None,
):
    files, with_text = scan_dataset(source_dir)
    if not files:
        raise ValueError("No supported audio files found.")
    if not project_name or str(project_name) == "None":
        raise ValueError("Create or select a Project first.")
    project = Path(projects_root) / _safe_name(project_name)
    audio_dir = project / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    manifest = project / "manifest.jsonl"
    rows = []
    from .training_data import plain_text, language_name
    language = language_name(language)
    for index, src in enumerate(files, 1):
        sidecar = src.with_suffix(".json")
        metadata = json.loads(sidecar.read_text(encoding="utf-8")) if sidecar.exists() else {}
        identity = metadata.get("speaker_id") or speaker_id
        if not identity:
            raise ValueError(f"{src.name}: supply speaker_id in its .json sidecar or explicitly declare one speaker for the folder.")
        text_path = src.with_suffix(".txt")
        transcript = text_path.read_text(encoding="utf-8").strip() if text_path.exists() else ""
        if not transcript and transcribe_missing:
            if asr_manager is None:
                raise ValueError(f"Missing transcript for {src.name} and ASR is disabled.")
            transcript, _ = asr_manager.transcribe(
                str(src), asr_model, asr_language, int(asr_batch_size)
            )
        if not transcript:
            log(f"[dataset] Skipping {src.name}: no transcript.")
            continue
        dst = audio_dir / f"{index:05d}_{src.stem}{src.suffix.lower()}"
        if src.resolve() != dst.resolve():
            shutil.copy2(src, dst)
        row = {"audio": str(dst.resolve()), "text": plain_text(transcript), "language": language,
               "speaker_id": str(identity)}
        for key in ("source_id", "segment_source_id", "global_spk_id", "start", "end"):
            if key in metadata:
                row[key] = metadata[key]
        rows.append(row)
        log(f"[dataset] Prepared {index}/{len(files)}: {src.name}")
    if not rows:
        raise ValueError("No usable samples were prepared.")
    with manifest.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    meta_path = project / "project.json"
    meta = {}
    if meta_path.is_file():
        try:
            loaded = json.loads(meta_path.read_text(encoding="utf-8-sig"))
            if isinstance(loaded, dict):
                meta.update(loaded)
        except Exception:
            pass
    meta.update({"name": project.name, "samples": len(rows), "language": language, "manifest": str(manifest)})
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return project, manifest, len(rows)



from .training_data import load_manifest
from .training_run import train_lora, TrainingOptions


def stop_training():
    STOP_EVENT.set()

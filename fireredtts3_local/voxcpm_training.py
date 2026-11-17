"""Pinned official VoxCPM2 LoRA training, isolated from the FireRed environment.

This module only prepares/audits configuration and supervises the upstream
trainer. It does not change the model, objective, precision, or generation code.
Pilot and full jobs start from the same published base in different directories;
an existing output is refused rather than silently invoking upstream auto-resume.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

SOURCE_URL = "https://github.com/OpenBMB/VoxCPM.git"
SOURCE_COMMIT = "300e4260b509066151d623dc9c0aa922cf82232b"
MODEL_ID = "openbmb/VoxCPM2"
MODEL_REVISION = "7936080b41163874f38610f46f438e2c0757f9fd"
DATASETS_VERSION = "3.6.0"
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / ".runtime/VoxCPM2"
DEFAULT_VENV = ROOT / ".venv-voxcpm2"
DEFAULT_MODEL = ROOT / "models/VoxCPM2"
SCHEMA = "voxcpm2-official-lora-run-v1"
MODEL_MARKER = ".voxcpm2-pinned-model.json"
MODEL_FILES = ("config.json", "model.safetensors", "audiovae.pth", "tokenizer.json",
               "tokenizer_config.json", "special_tokens_map.json")
LORA = {"enable_lm": True, "enable_dit": True, "enable_proj": False,
        "r": 32, "alpha": 32, "dropout": 0.0}


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def read_yaml(path):
    import yaml
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Recipe must be a YAML mapping")
    return value


def write_yaml(path, value):
    import yaml
    Path(path).write_text(yaml.safe_dump(value, allow_unicode=True, sort_keys=False), encoding="utf-8")


def run_checked(command, *, cwd=None, capture=False):
    return subprocess.run([str(part) for part in command], cwd=cwd, check=True,
                          text=True, capture_output=capture)


def verify_source(source):
    source = Path(source).resolve()
    current = run_checked(["git", "-C", source, "rev-parse", "HEAD"], capture=True).stdout.strip()
    if current != SOURCE_COMMIT:
        raise ValueError(f"VoxCPM source must be pinned to {SOURCE_COMMIT}; run setup")
    changed = run_checked(["git", "-C", source, "status", "--porcelain", "--untracked-files=no"],
                          capture=True).stdout.strip()
    if changed:
        raise ValueError("Pinned VoxCPM source has tracked changes; use an unchanged checkout")
    if not (source / "scripts/train_voxcpm_finetune.py").is_file():
        raise ValueError("Pinned official trainer is missing")
    return source


def ensure_source(source):
    """Retry an interrupted initial fetch without replacing an existing checkout."""
    source = Path(source).resolve()
    if not (source / ".git").is_dir():
        if source.exists() and any(source.iterdir()):
            raise ValueError("Refusing to initialize VoxCPM in a nonempty non-git directory")
        source.parent.mkdir(parents=True, exist_ok=True)
        run_checked(["git", "init", source])
        run_checked(["git", "-C", source, "remote", "add", "origin", SOURCE_URL])
    head = subprocess.run(["git", "-C", str(source), "rev-parse", "--verify", "HEAD"],
                          text=True, capture_output=True)
    if head.returncode:
        origin = run_checked(["git", "-C", source, "remote", "get-url", "origin"], capture=True).stdout.strip()
        status = run_checked(["git", "-C", source, "status", "--porcelain"], capture=True).stdout.strip()
        if origin != SOURCE_URL or status:
            raise ValueError("Uninitialized source directory is not an empty official VoxCPM checkout")
        run_checked(["git", "-C", source, "fetch", "--depth=1", "origin", SOURCE_COMMIT])
        run_checked(["git", "-C", source, "checkout", "--detach", "FETCH_HEAD"])
    return verify_source(source)


def verify_model(model, *, hashes=True):
    model = Path(model).resolve()
    marker = model / MODEL_MARKER
    if not marker.is_file():
        raise ValueError("Model has no pinned download receipt; run voxcpm_training setup")
    receipt = json.loads(marker.read_text(encoding="utf-8"))
    if (receipt.get("repo_id"), receipt.get("revision"), receipt.get("complete")) != (
            MODEL_ID, MODEL_REVISION, True):
        raise ValueError("Model receipt does not identify the required complete VoxCPM2 base")
    if set(receipt.get("files", {})) != set(MODEL_FILES):
        raise ValueError("Pinned model receipt is incomplete")
    for name in MODEL_FILES:
        path = model / name
        if not path.is_file() or (hashes and digest(path) != receipt["files"][name]):
            raise ValueError(f"Pinned model file changed or is missing: {name}")
    config = json.loads((model / "config.json").read_text(encoding="utf-8"))
    if (config.get("architecture"), config.get("patch_size"), config.get("feat_dim")) != (
            "voxcpm2", 4, 64):
        raise ValueError("Unexpected architecture in pinned VoxCPM2 config")
    return config


def download_model(model):
    """Called inside the separate environment; credentials come from HF's cache/env."""
    from huggingface_hub import snapshot_download
    model = Path(model).resolve()
    marker = model / MODEL_MARKER
    if model.exists() and any(model.iterdir()):
        if not marker.is_file():
            raise ValueError("Refusing to replace a nonempty model directory without our download receipt")
        receipt = json.loads(marker.read_text(encoding="utf-8"))
        if (receipt.get("repo_id"), receipt.get("revision")) != (MODEL_ID, MODEL_REVISION):
            raise ValueError("Model directory belongs to a different download")
    model.mkdir(parents=True, exist_ok=True)
    write_json(marker, {"repo_id": MODEL_ID, "revision": MODEL_REVISION, "complete": False})
    try:
        snapshot_download(repo_id=MODEL_ID, revision=MODEL_REVISION, local_dir=model,
                          allow_patterns=list(MODEL_FILES), max_workers=2)
    except Exception:
        raise RuntimeError("Pinned model download failed; retry setup. Credentials and response URLs are not logged.") from None
    receipt = {"repo_id": MODEL_ID, "revision": MODEL_REVISION, "complete": True,
               "files": {name: digest(model / name) for name in MODEL_FILES}}
    write_json(marker, receipt)
    return {"model": str(model), "revision": MODEL_REVISION, "complete": True}


def setup(source=DEFAULT_SOURCE, venv=DEFAULT_VENV, model=DEFAULT_MODEL, *, no_download=False):
    """Install the official checkout in a dedicated venv; never install FireRed here."""
    source, venv, model = map(lambda path: Path(path).resolve(), (source, venv, model))
    uv = shutil.which("uv")
    if not uv:
        raise ValueError("uv is required for setup; install uv first (https://docs.astral.sh/uv/)")
    ensure_source(source)
    if not (venv / "pyvenv.cfg").is_file():
        if venv.exists() and any(venv.iterdir()):
            raise ValueError("Refusing to replace a nonempty non-venv directory")
        run_checked([uv, "venv", "--python", "3.12", venv])
    python = venv / "bin/python"
    constraints = venv / "voxcpm-constraints.txt"
    constraints.write_text(f"torch==2.8.0\ntorchaudio==2.8.0\ntorchcodec==0.7.0\ndatasets=={DATASETS_VERSION}\n",
                           encoding="utf-8")
    run_checked([uv, "pip", "install", "--python", python, "--index-url",
                 "https://download.pytorch.org/whl/cu128", "torch==2.8.0", "torchaudio==2.8.0"])
    run_checked([uv, "pip", "install", "--python", python, "--constraint", constraints,
                 "-e", source, "torchcodec==0.7.0", "tensorboardX", "PyYAML", "silero-vad==6.2.3"])
    # Upstream declares torchcodec as a dependency, but its datasets 3.x audio
    # loader and native VoxCPM2 reference path use SoundFile/librosa. Importing
    # unused TorchCodec here incorrectly requires FFmpeg 4-7 shared libraries
    # even on hosts whose actual training/inference audio backend works.
    run_checked([python, "-m", "fireredtts3_local.voxcpm_training", "check-runtime"], cwd=ROOT)
    frozen = run_checked([uv, "pip", "freeze", "--python", python], capture=True).stdout
    (venv / "voxcpm-installed.txt").write_text(frozen, encoding="utf-8")
    write_json(venv / "voxcpm-environment.json", {"schema": SCHEMA, "source": str(source),
               "source_commit": SOURCE_COMMIT, "python": str(python), "model": str(model),
               "model_revision": MODEL_REVISION})
    if not no_download:
        run_checked([python, "-m", "fireredtts3_local.voxcpm_training", "_download", str(model)], cwd=ROOT)
    return {"python": str(python), "source": str(source), "source_commit": SOURCE_COMMIT,
            "model": str(model), "model_revision": MODEL_REVISION,
            "next": f"{python} -m fireredtts3_local.voxcpm_training build-recipe --help"}


def check_audio_runtime():
    """Exercise the actual datasets/native audio paths, without weights or GPU."""
    import tempfile
    import numpy as np
    import soundfile as sf
    import librosa
    import torch
    import torchaudio
    import yaml  # noqa: F401
    import tensorboardX  # noqa: F401
    from datasets import Audio, Dataset, __version__ as datasets_version
    from voxcpm.model import VoxCPM2Model  # noqa: F401

    if datasets_version != DATASETS_VERSION:
        raise ValueError(f"Expected datasets=={DATASETS_VERSION}, got {datasets_version}; rerun setup")
    with tempfile.TemporaryDirectory(prefix="voxcpm-audio-check-") as temporary:
        paths = []
        for name, rate, frequency in (("target", 24000, 440), ("reference", 48000, 330)):
            path = Path(temporary) / f"{name}.wav"
            samples = (.1 * np.sin(2 * np.pi * frequency * np.arange(rate // 4) / rate)).astype("float32")
            sf.write(path, samples, rate, subtype="FLOAT")
            paths.append(str(path))
        dataset = Dataset.from_dict({"audio": paths[:1], "ref_audio": paths[1:]})
        for field in ("audio", "ref_audio"):
            dataset = dataset.cast_column(field, Audio(sampling_rate=16000))
        decoded = dataset[0]
        for field in ("audio", "ref_audio"):
            audio = decoded[field]
            wave = np.asarray(audio["array"])
            if (audio["sampling_rate"] != 16000 or wave.shape != (4000,)
                    or not np.isfinite(wave).all() or not np.any(wave)):
                raise RuntimeError(f"Official datasets audio backend failed to decode/resample {field}")
        # Native VoxCPM2 uses librosa.load for the voice/prompt reference.
        reference, rate = librosa.load(paths[1], sr=16000, mono=True)
        if rate != 16000 or reference.shape != (4000,) or not np.isfinite(reference).all():
            raise RuntimeError("Native VoxCPM2 reference decoding failed")
        np.testing.assert_allclose(reference, decoded["ref_audio"]["array"], atol=1e-6)
    return {"audio_runtime": "ok", "torch": torch.__version__, "cuda": torch.version.cuda,
            "torchaudio": torchaudio.__version__, "datasets": datasets_version,
            "audio_backend": "datasets.Audio / SoundFile / librosa", "sample_rate": 16000,
            "checked": ["FLOAT WAV target", "FLOAT WAV reference", "native reference resampling"],
            "torchcodec_required_for_this_workflow": False}


def prepared_audit(dataset):
    from .voxcpm_data import audit
    report = audit(Path(dataset))
    if report.get("complete") is not True:
        raise ValueError("Prepared data audit is not complete")
    return report


def tokenizer_for(model, source=None):
    import voxcpm
    from transformers import LlamaTokenizerFast
    from voxcpm.model.utils import mask_multichar_chinese_tokens
    if source is not None and not Path(voxcpm.__file__).resolve().is_relative_to(Path(source).resolve() / "src"):
        raise ValueError("This Python imports a different VoxCPM checkout; use .venv-voxcpm2/bin/python")
    return mask_multichar_chinese_tokens(LlamaTokenizerFast.from_pretrained(str(model), local_files_only=True))


def sequence_audit(dataset, tokenizer, *, context_limit, batch_size, max_batch_tokens):
    """Check *every* target/reference: no silent upstream length drop or truncation."""
    import soundfile as sf
    dataset = Path(dataset)
    cache, result = {}, {}
    official_limit = max_batch_tokens // batch_size if max_batch_tokens else context_limit
    limit = min(context_limit, official_limit)
    if limit < 1:
        raise ValueError("Token budget is smaller than one batch")
    for split in ("train", "validation"):
        rows = [json.loads(line) for line in (dataset / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()]
        if not rows:
            raise ValueError(f"Empty {split} manifest")
        maximum, hours = 0, 0.0
        for index, row in enumerate(rows):
            if not isinstance(row.get("text"), str) or not row["text"].strip():
                raise ValueError(f"Empty transcript: {split} row {index + 1}")
            if not row.get("ref_audio") or Path(row["audio"]).resolve() == Path(row["ref_audio"]).resolve():
                raise ValueError(f"Distinct ref_audio is required: {split} row {index + 1}")
            durations = []
            for field, duration_field in (("audio", "duration"), ("ref_audio", "ref_duration")):
                path = Path(row[field])
                if not path.is_absolute():
                    path = dataset / path
                path = path.resolve()
                if path not in cache:
                    info = sf.info(path)
                    if info.channels != 1 or info.frames <= 0:
                        raise ValueError(f"Audio must be nonempty mono: {path}")
                    cache[path] = info.duration
                actual = cache[path]
                stated = float(row.get(duration_field, float("nan")))
                if not math.isfinite(stated) or stated <= 0 or abs(actual - stated) > .02:
                    raise ValueError(f"Invalid duration metadata: {split} row {index + 1} {field}")
                # Check both actual length and the exact upstream metadata estimate.
                durations.append(max(actual, stated))
            length = len(tokenizer(row["text"])) + sum(math.ceil(math.ceil(d * 25) / 4) for d in durations) + 4
            if length > limit:
                raise ValueError(f"Whole {split} row {index + 1} needs {length} tokens; limit is {limit}. "
                                 "Nothing was truncated or skipped. Reduce batch size or increase max-batch-tokens "
                                 "within model context; over-context recordings need a separate data decision.")
            maximum = max(maximum, length)
            hours += durations[0] / 3600
        result[split] = {"targets": len(rows), "hours": hours, "max_sequence_tokens": maximum}
    result.update(context_limit=context_limit, upstream_filter_limit=official_limit, all_targets_fit=True)
    return result


def build_recipe(dataset, output, output_dir, *, source=DEFAULT_SOURCE, model=DEFAULT_MODEL,
                 python=None, epochs=1, batch_size=1, grad_accum_steps=8, learning_rate=1e-4,
                 max_batch_tokens=8192, num_workers=4):
    dataset, output, output_dir, source, model = map(lambda p: Path(p).resolve(),
                                                     (dataset, output, output_dir, source, model))
    if any(value < 1 for value in (epochs, batch_size, grad_accum_steps, max_batch_tokens)):
        raise ValueError("Epochs, batch size, accumulation and token budget must be positive")
    if not math.isfinite(learning_rate) or not 0 < learning_rate <= 1e-3:
        raise ValueError("Learning rate must be finite, positive and <= 1e-3")
    if num_workers < 0:
        raise ValueError("num_workers cannot be negative")
    if output.exists() or output.with_suffix(".audit.json").exists():
        raise ValueError("Recipe/audit already exists; choose a new path for an immutable experiment")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("Training output must be fresh; official automatic resume is not used")
    verify_source(source)
    config = verify_model(model)
    report = prepared_audit(dataset)
    lengths = sequence_audit(dataset, tokenizer_for(model, source), context_limit=int(config["max_length"]),
                             batch_size=batch_size, max_batch_tokens=max_batch_tokens)
    targets = lengths["train"]["targets"]
    # Official DataLoader uses drop_last=True. Require exact batches, not silent
    # per-epoch discarded targets. Default batch=1 supports any prepared count.
    if targets % batch_size:
        raise ValueError(f"Official trainer uses drop_last=True: {targets} targets is not divisible by "
                         f"batch_size={batch_size}. Use batch_size=1 or a divisor; no targets were dropped.")
    updates = math.ceil(epochs * (targets // batch_size) / grad_accum_steps)
    warmup = min(100, max(1, updates // 10)) if updates > 1 else 0
    recipe = {"pretrained_path": str(model), "train_manifest": str(dataset / "train.jsonl"),
              "val_manifest": str(dataset / "validation.jsonl"), "sample_rate": 16000,
              "out_sample_rate": 48000, "batch_size": batch_size, "grad_accum_steps": grad_accum_steps,
              "num_workers": num_workers, "preprocessing_num_workers": max(1, num_workers),
              "num_iters": updates, "log_interval": 1, "valid_interval": min(100, updates),
              "save_interval": min(100, updates), "learning_rate": learning_rate,
              "weight_decay": .01, "warmup_steps": warmup, "max_steps": updates,
              "max_batch_tokens": max_batch_tokens, "max_grad_norm": 1., "save_path": str(output_dir),
              "tensorboard": str(output_dir / "logs"), "lambdas": {"loss/diff": 1., "loss/stop": 1.},
              "lora": dict(LORA)}
    output.parent.mkdir(parents=True, exist_ok=True)
    write_yaml(output, recipe)
    manifest_paths = [dataset / name for name in report["manifests"]]
    audit = {"schema": SCHEMA, "source": str(source), "source_commit": SOURCE_COMMIT,
             # Resolving the interpreter symlink escapes its venv and loses the
             # isolated package installation. Keep its absolute lexical path.
             "python": os.path.abspath(python or sys.executable), "model_revision": MODEL_REVISION,
             "model_receipt_sha256": digest(model / MODEL_MARKER), "dataset": str(dataset),
             "dataset_report_sha256": digest(dataset / "report.json"),
             "manifests": {str(p): digest(p) for p in manifest_paths}, "sequence_audit": lengths,
             "recipe_sha256": digest(output), "requested_epochs": epochs, "updates": updates,
             "target_draws": updates * batch_size * grad_accum_steps,
             "official_validation": {"max_batches_per_call": 10, "tensorboard_generated_samples": 2,
                                     "tensorboard_generation_uses_reference": False,
                                     "covers_full_validation_manifest": lengths["validation"]["targets"] <= 10 * batch_size},
             "note": "Official shuffled DataLoader; rounded-up updates may begin the next epoch. "
                     "Fresh base, official loss. Upstream validation is capped at 10 batches; "
                     "zero-based step labels are recorded after an update. "
                     "Independent whole-utterance cloning/listening validation is required before release."}
    write_json(output.with_suffix(".audit.json"), audit)
    return {"recipe": str(output), "audit": str(output.with_suffix(".audit.json")), **lengths,
            "updates": updates, "target_draws": audit["target_draws"], "output": str(output_dir)}


def verify_recipe(recipe_path, audit_path=None, *, check_data=True):
    recipe_path = Path(recipe_path).resolve()
    audit_path = Path(audit_path or recipe_path.with_suffix(".audit.json"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("schema") != SCHEMA or audit.get("recipe_sha256") != digest(recipe_path):
        raise ValueError("Recipe changed after audit; build a new recipe")
    recipe = read_yaml(recipe_path)
    if recipe.get("lora") != LORA or recipe.get("sample_rate") != 16000 or recipe.get("out_sample_rate") != 48000:
        raise ValueError("Unexpected official LoRA/encoder configuration")
    if audit.get("source_commit") != SOURCE_COMMIT or audit.get("model_revision") != MODEL_REVISION:
        raise ValueError("Recipe provenance does not match pinned source and base")
    verify_source(audit["source"])
    if not Path(audit["python"]).is_file():
        raise ValueError("Recipe Python is missing; build with the isolated VoxCPM environment")
    verify_model(recipe["pretrained_path"])
    if digest(Path(recipe["pretrained_path"]) / MODEL_MARKER) != audit["model_receipt_sha256"]:
        raise ValueError("Model receipt changed after recipe build")
    dataset = Path(audit["dataset"])
    if digest(dataset / "report.json") != audit["dataset_report_sha256"]:
        raise ValueError("Prepared data report changed after recipe build")
    for path, expected in audit["manifests"].items():
        if digest(path) != expected:
            raise ValueError("Prepared manifests changed after recipe build")
    if check_data:
        prepared_audit(dataset)
    return recipe, audit


def check_training_runtime(python, source):
    """Fail before loading 2B weights if the selected venv/GPU is not usable."""
    script = (
        "import importlib.util,pathlib,sys,torch\n"
        "spec=importlib.util.find_spec('voxcpm')\n"
        "expected=pathlib.Path(sys.argv[1]).resolve()/'src'\n"
        "if spec is None or not spec.origin or not pathlib.Path(spec.origin).resolve().is_relative_to(expected):\n"
        "    raise SystemExit('Selected Python does not import the pinned VoxCPM checkout; run setup')\n"
        "if not torch.cuda.is_available():\n"
        "    raise SystemExit('CUDA GPU is unavailable; no training started')\n"
        "if not torch.cuda.is_bf16_supported():\n"
        "    raise SystemExit('This official BF16 recipe requires a BF16-capable CUDA GPU')\n"
        "print('[voxcpm] GPU:',torch.cuda.get_device_name(0),'torch:',torch.__version__,flush=True)\n"
    )
    run_checked([python, "-c", script, str(source)])


def launch(recipe_path, job_dir, *, pilot_steps=None):
    import fcntl
    from .training_job import now, write_json as atomic_json
    recipe_path, job = Path(recipe_path).resolve(), Path(job_dir).resolve()
    # Rehash the large audio corpus in the detached worker before training,
    # rather than twice (also while the initiating SSH shell is waiting).
    recipe, audit = verify_recipe(recipe_path, check_data=False)
    if pilot_steps is not None:
        if not 1 <= pilot_steps < recipe["num_iters"]:
            raise ValueError("Pilot steps must be positive and smaller than the full run")
        recipe["num_iters"] = pilot_steps
        recipe["save_path"] += f"-pilot-{pilot_steps}"
        recipe["tensorboard"] = str(Path(recipe["save_path"]) / "logs")
    output = Path(recipe["save_path"]).resolve()
    locks = ROOT / ".runtime/voxcpm-job-locks"
    locks.mkdir(parents=True, exist_ok=True)
    lock_path = locks / (hashlib.sha256(str(output).encode()).hexdigest() + ".lock")
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError(f"A background job already owns {output}") from exc
        if output.exists() and any(output.iterdir()):
            raise ValueError("Training output is not empty; choose a new run. Upstream auto-resume is deliberately disabled.")
        job.mkdir(parents=True, exist_ok=False)
        lock.seek(0)
        lock.truncate()
        lock.write(str(job))
        lock.flush()
        write_yaml(job / "recipe.yaml", recipe)
        audit.update(recipe_sha256=digest(job / "recipe.yaml"), pilot_steps=pilot_steps,
                     full_recipe_sha256=digest(recipe_path))
        write_json(job / "recipe.audit.json", audit)
        state = {"status": "starting", "created_at": now(), "cwd": str(ROOT),
                 "python": audit["python"], "source_recipe": str(recipe_path), "output_dir": str(output),
                 "lock_path": str(lock_path), "log": str(job / "job.log"), "pilot_steps": pilot_steps,
                 "source_commit": SOURCE_COMMIT, "model_revision": MODEL_REVISION}
        atomic_json(job / "status.json", state)
        command = [sys.executable, "-u", "-m", "fireredtts3_local.voxcpm_training",
                   "_worker", str(job), str(lock.fileno())]
        try:
            with (job / "job.log").open("ab", buffering=0) as log:
                process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
                                           stderr=subprocess.STDOUT, start_new_session=True,
                                           close_fds=True, pass_fds=(lock.fileno(),))
        except BaseException:
            state.update(status="failed", finished_at=now(), error="Could not launch worker")
            atomic_json(job / "status.json", state)
            raise
    return {"pid": process.pid, "job_dir": str(job), "log": str(job / "job.log"),
            "output_dir": str(output), "pilot": pilot_steps is not None,
            "note": "Pilot is a separate smoke/performance run. Full training starts fresh from base."}


def worker(job, lock_fd):
    from .training_job import now, run_commands, write_json as atomic_json
    job = Path(job).resolve()
    state = json.loads((job / "status.json").read_text(encoding="utf-8"))
    os.chdir(state["cwd"])
    try:
        recipe, audit = verify_recipe(job / "recipe.yaml", job / "recipe.audit.json")
        output = Path(recipe["save_path"])
        if output.exists() and any(output.iterdir()):
            raise ValueError("Output became nonempty before launch; refusing implicit resume")
        check_training_runtime(audit["python"], audit["source"])
        trainer = Path(audit["source"]) / "scripts/train_voxcpm_finetune.py"
        command = [audit["python"], "-u", str(trainer), "--config_path", str(job / "recipe.yaml")]
        print(f"[voxcpm] Official source {SOURCE_COMMIT}; base {MODEL_REVISION}", flush=True)
        print("[voxcpm] Fresh LoRA; automatic resume disabled. Evaluate complete generated speech before release.", flush=True)
        return run_commands(job, state, [("official-train", command)], lock_fd)
    except BaseException as exc:
        state.update(status="failed", finished_at=now(), error=str(exc))
        atomic_json(job / "status.json", state)
        raise
    finally:
        os.close(lock_fd)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    install = commands.add_parser("setup", help="Install pinned upstream and download pinned base in an isolated venv")
    install.add_argument("--source", default=str(DEFAULT_SOURCE))
    install.add_argument("--venv", default=str(DEFAULT_VENV))
    install.add_argument("--model", default=str(DEFAULT_MODEL))
    install.add_argument("--no-download", action="store_true")
    commands.add_parser("check-runtime", help="Check actual audio decoding/resampling without downloading weights or using a GPU")
    build = commands.add_parser("build-recipe", help="Audit all whole recordings and create a fresh official LoRA recipe")
    build.add_argument("--dataset", default="training/datasets/voxcpm2-ru")
    build.add_argument("--output", default="training/recipes/voxcpm2-ru.yaml")
    build.add_argument("--output-dir", default="training/outputs/voxcpm2-ru-lora")
    build.add_argument("--source", default=str(DEFAULT_SOURCE))
    build.add_argument("--model", default=str(DEFAULT_MODEL))
    build.add_argument("--epochs", type=int, default=1)
    build.add_argument("--batch-size", type=int, default=1)
    build.add_argument("--grad-accum-steps", type=int, default=8)
    build.add_argument("--learning-rate", type=float, default=1e-4)
    build.add_argument("--max-batch-tokens", type=int, default=8192)
    build.add_argument("--num-workers", type=int, default=4)
    start = commands.add_parser("start", help="Start a detached fresh-base job; no implicit resume")
    start.add_argument("recipe")
    start.add_argument("--job-dir", required=True)
    start.add_argument("--pilot-steps", type=int)
    for action in ("status", "stop"):
        sub = commands.add_parser(action)
        sub.add_argument("job_dir")
    check = commands.add_parser("audit", help="Recheck immutable recipe, source, model and prepared data")
    check.add_argument("recipe")
    work = commands.add_parser("_worker", help=argparse.SUPPRESS)
    work.add_argument("job_dir")
    work.add_argument("lock_fd", type=int)
    dl = commands.add_parser("_download", help=argparse.SUPPRESS)
    dl.add_argument("model")
    args = parser.parse_args(argv)
    if args.command == "setup":
        result = setup(args.source, args.venv, args.model, no_download=args.no_download)
    elif args.command == "check-runtime":
        result = check_audio_runtime()
    elif args.command == "build-recipe":
        options = vars(args).copy()
        for key in ("command", "dataset", "output", "output_dir"):
            options.pop(key)
        result = build_recipe(args.dataset, args.output, args.output_dir, **options)
    elif args.command == "start":
        if os.name != "posix":
            parser.error("Detached jobs require Linux/POSIX")
        result = launch(args.recipe, args.job_dir, pilot_steps=args.pilot_steps)
    elif args.command in {"status", "stop"}:
        from .training_job import read_status
        job = Path(args.job_dir).resolve()
        result = read_status(job)
        if args.command == "stop":
            if not result["active"]:
                parser.error("Job is not active")
            (job / "STOP").touch()
            result = {"stop_requested": True, "job_dir": str(job)}
    elif args.command == "audit":
        _, result = verify_recipe(args.recipe)
    elif args.command == "_download":
        result = download_model(args.model)
    else:
        return worker(args.job_dir, args.lock_fd)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Export the final Russian VoxCPM2 checkpoint as a standalone model for serving."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil
import tempfile

from filelock import FileLock
from huggingface_hub import HfApi, hf_hub_download
from safetensors import safe_open
from safetensors.torch import load_file, save_file
import torch

from .release import matches_hub_file, remote_matches
from .voxcpm_eval import adapter_info, full_checkpoint_info
from .voxcpm_training import (DEFAULT_MODEL, DEFAULT_SOURCE, LORA, MODEL_FILES, MODEL_ID, MODEL_MARKER,
                              MODEL_REVISION, ROOT, digest, verify_model, verify_source, write_json)

CHECKPOINT = ROOT / "training/outputs/voxcpm2-ru-lora/latest"
OUTPUT = ROOT / "training/exports/VoxCPM2-Russian"
DESTINATION = "shustov-te/VoxCPM2-Russian"
STEP = 9114
WEIGHTS = "model.safetensors"
TOKENIZER_CODE = "tokenization_voxcpm2.py"
TOKENIZER_SHA = "84489ea32b6ee0cae22ed5480cacb6df85c46624c3119be9a2021c3649a12729"
PAYLOAD = {*MODEL_FILES, TOKENIZER_CODE, "README.md", "UPSTREAM_MODEL_CARD.md", "LICENSE"}
RECEIPT = "export_provenance.json"


def merge_weights(base, adapter, config, output):
    """Operate on CPU, keep the original state-dict layout, validate every LoRA pair."""
    if any(config.get(k) != v for k, v in LORA.items()):
        raise ValueError("Unexpected LoRA profile for the final Russian run")
    weights = load_file(base, device="cpu")
    lora = load_file(adapter, device="cpu")
    targets = set()
    for root, field in [("base_lm", "target_modules_lm"), ("residual_lm", "target_modules_lm"),
                        ("feat_decoder.estimator", "target_modules_dit")]:
        names = set(config[field])
        group = {k.removesuffix(".weight") for k in weights if k.startswith(root + ".")
                 and k.endswith(".weight") and k.split(".")[-2] in names}
        if not group or {p.split(".")[-1] for p in group} != names:
            raise ValueError(f"Base is missing LoRA targets under {root}")
        targets.update(group)
    if set(lora) != {f"{p}.lora_{s}" for p in targets for s in ("A", "B")}:
        raise ValueError("Missing or unexpected LoRA tensors; refusing partial merge")
    rank, scale = config["r"], config["alpha"] / config["r"]
    changed = 0
    for name in sorted(targets):
        key = name + ".weight"
        weight = weights[key]
        a, b = lora[name + ".lora_A"], lora[name + ".lora_B"]
        if (weight.ndim != 2 or a.shape != (rank, weight.shape[1]) or b.shape != (weight.shape[0], rank)
                or not all(t.is_floating_point() for t in (weight, a, b))):
            raise ValueError(f"Incompatible LoRA shapes or dtypes: {name}")
        merged = torch.addmm(weight.float(), b.float(), a.float(), alpha=scale).to(weight.dtype)
        if not torch.isfinite(merged).all():
            raise ValueError(f"Non-finite merged weight: {name}")
        changed += int(not torch.equal(weight, merged))
        weights[key] = merged
    if not changed:
        raise ValueError("Adapter made no representable change to the base")
    with safe_open(base, framework="pt") as original:
        metadata = original.metadata()
    save_file(weights, output, metadata=metadata)
    with safe_open(output, framework="pt") as result:
        if set(result.keys()) != set(weights) or any(
            not torch.equal(result.get_tensor(k), value) for k, value in weights.items()
        ):
            raise ValueError("Saved model differs from the verified merged state")
    return {"target_layers": len(targets), "changed_layers": changed}


def prepare(output, *, full_checkpoint=None):
    verify_model(DEFAULT_MODEL)
    source = verify_source(DEFAULT_SOURCE)
    base_receipt = json.loads((DEFAULT_MODEL / MODEL_MARKER).read_text())
    identity = {"base": MODEL_ID, "base_revision": MODEL_REVISION}
    if full_checkpoint is None:
        adapter = adapter_info(CHECKPOINT, DEFAULT_MODEL.resolve())
        if json.loads((CHECKPOINT / "training_state.json").read_text())["step"] != STEP:
            raise ValueError(f"Expected final checkpoint step {STEP}")
        if Path(adapter["weights"]).name != "lora_weights.safetensors":
            raise ValueError("Expected native safetensors adapter")
        identity.update(step=STEP, adapter=adapter["files"])
    else:
        checkpoint, step, trained = full_checkpoint_info(full_checkpoint, DEFAULT_MODEL, base_receipt["files"])
        identity.update(step=step, training_mode="full", weights_sha256=trained[WEIGHTS])
    if output.exists():
        receipt = json.loads((output / RECEIPT).read_text())
        if (receipt["source"] != identity or set(receipt["files"]) != PAYLOAD
                or {p.name for p in output.iterdir()} != PAYLOAD | {RECEIPT}
                or any(digest(output / k) != receipt["files"][k] for k in PAYLOAD)):
            raise ValueError("Existing export differs; use a fresh --output directory")
        return receipt

    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".voxcpm-export-") as tmp:
        staging = Path(tmp) / "model"
        staging.mkdir()
        for name in sorted(set(MODEL_FILES) - {WEIGHTS}):
            shutil.copyfile(DEFAULT_MODEL / name, staging / name)
        for original, dest in [(TOKENIZER_CODE, TOKENIZER_CODE), ("README.md", "UPSTREAM_MODEL_CARD.md")]:
            path = hf_hub_download(MODEL_ID, original, revision=MODEL_REVISION)
            shutil.copyfile(path, staging / dest)
        if digest(staging / TOKENIZER_CODE) != TOKENIZER_SHA:
            raise ValueError("Incorrect upstream tokenizer implementation")
        shutil.copyfile(source / "LICENSE", staging / "LICENSE")
        receipt = {"source": identity}
        if full_checkpoint is None:
            print("Merging LM and DiT weights on CPU ...", flush=True)
            stats = merge_weights(DEFAULT_MODEL / WEIGHTS, adapter["weights"], adapter["config"], staging / WEIGHTS)
            if any(digest(CHECKPOINT / k) != h for k, h in adapter["files"].items()):
                raise ValueError("Adapter changed during export")
            receipt["merge"] = stats
            training = f"Russian LoRA merged into {stats['target_layers']} layers after {STEP} updates"
        else:
            print(f"Copying full checkpoint, step {step} ...", flush=True)
            shutil.copyfile(checkpoint / WEIGHTS, staging / WEIGHTS)
            training = f"Full fine-tuning from the original base for {step} optimizer updates"
        files = {k: digest(staging / k) for k in sorted(PAYLOAD - {"README.md"})}
        if full_checkpoint is not None and files[WEIGHTS] != trained[WEIGHTS]:
            raise ValueError("Exported weights differ from the full checkpoint")
        if any(files[k] != base_receipt["files"][k] for k in set(MODEL_FILES) - {WEIGHTS}):
            raise ValueError("Copied base assets differ from the pinned model")
        (staging / "README.md").write_text(
            f"---\nlanguage: [ru]\nbase_model: {MODEL_ID}\npipeline_tag: text-to-speech\n"
            "library_name: voxcpm\nlicense: apache-2.0\n---\n\n# VoxCPM2 Russian\n\n"
            f"Complete VoxCPM2 checkpoint derived from {MODEL_ID} at `{MODEL_REVISION}`. "
            f"{training} on Dialogs, "
            "Russian Common Voice Spontaneous and Scripted. AudioVAE and tokenizer are from the base. "
            "Use as a standalone VoxCPM2 model, including with vLLM-Omni. "
            "See UPSTREAM_MODEL_CARD.md for base model attribution and usage.\n", encoding="utf-8")
        files["README.md"] = digest(staging / "README.md")
        receipt["files"] = files
        write_json(staging / RECEIPT, receipt)
        staging.rename(output)
    return receipt


def upload_private(root, api, *, expected_parent=None):
    if expected_parent is None:
        api.create_repo(DESTINATION, repo_type="model", private=True, exist_ok=True)
    elif not re.fullmatch(r"[0-9a-f]{40}", expected_parent):
        raise ValueError("Expected the complete 40-character parent commit SHA")
    head = api.model_info(DESTINATION, revision="main", files_metadata=True)
    names = PAYLOAD | {RECEIPT}
    if head.private is not True:
        raise ValueError("Destination is not private; no upload")
    inventory = {f.rfilename: f for f in head.siblings if f.rfilename != ".gitattributes"}
    changes = names
    if expected_parent is not None:
        if inventory.keys() != names:
            raise ValueError("Previous private release has an unexpected file inventory")
        changes = {name for name in names if not matches_hub_file(root / name, inventory[name])}
        if not changes:
            return head.sha
        if head.sha != expected_parent:
            raise ValueError(f"Private main is at {head.sha}; expected {expected_parent}. No files uploaded")
        if changes - {WEIGHTS, "README.md", RECEIPT}:
            raise ValueError("Tokenizer, AudioVAE or other base assets differ from the previous release")
    elif inventory:
        if not remote_matches(head, root, names):
            raise ValueError("Destination already contains a different release; no files overwritten")
        return head.sha
    step = json.loads((root / RECEIPT).read_text())["source"]["step"]
    commit = api.upload_folder(repo_id=DESTINATION, folder_path=root, repo_type="model",
                               revision="main", allow_patterns=sorted(changes), parent_commit=head.sha,
                               commit_message=f"Publish complete Russian VoxCPM2, step {step}").oid
    remote = api.model_info(DESTINATION, revision=commit, files_metadata=True)
    if remote.private is not True or not remote_matches(remote, root, names):
        raise ValueError(f"Private release verification failed at {commit}")
    return commit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--full-checkpoint", type=Path, help="Official full fine-tuning checkpoint directory")
    parser.add_argument("--expected-parent", help="Current private main commit for a full checkpoint update")
    parser.add_argument("--publish", action="store_true", help=f"Upload to private {DESTINATION}")
    args = parser.parse_args()
    if args.expected_parent and not args.full_checkpoint:
        parser.error("--expected-parent is for a full checkpoint update")
    if args.full_checkpoint and args.publish and not args.expected_parent:
        parser.error("--publish with --full-checkpoint requires --expected-parent")
    default_output = (OUTPUT.with_name(f"{OUTPUT.name}-full-{args.full_checkpoint.resolve().name}")
                      if args.full_checkpoint else OUTPUT)
    output = (args.output or default_output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output) + ".lock"):
        receipt = prepare(output, full_checkpoint=args.full_checkpoint)
        print(f"Complete model: {output} (step {receipt['source']['step']})", flush=True)
        if args.publish:
            commit = upload_private(output, HfApi(token=True), expected_parent=args.expected_parent)
            print(f"Verified private release: https://huggingface.co/{DESTINATION}/tree/{commit}")


if __name__ == "__main__":
    main()

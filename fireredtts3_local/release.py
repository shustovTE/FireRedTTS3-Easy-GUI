"""Build a verified CKermit-compatible checkpoint and publish it to the token owner."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile

from .export import TOKENIZER_FILES, export_merged
from .hub_download import download_files
from .training_data import file_digest

TEMPLATE_REPO = "CKermit/FireRedTTS3-Base"
TEMPLATE_REVISION = "53f8b59c3252f15d539a9205fbcab3e334a36b9f"
OMNI_REVISION = "718b51d52a2cffe2218f9c6a1ae076756eb8a505"
OMNI_PR = "https://github.com/sgl-project/sglang-omni/pull/3005"
WEIGHTS = {"model.safetensors": "fireredtts3_base/model.safetensors",
           "redae/model.safetensors": "redae/model.safetensors",
           "campp/campplus_voxceleb.bin": "campp/campplus_voxceleb.bin"}
MANIFEST = "release_manifest.json"
PRODUCER = "fireredtts3-easy-gui"


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def git_blob_digest(path):
    path = Path(path)
    digest = hashlib.sha1(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def matches_hub_file(path, metadata):
    path = Path(path)
    if not path.is_file() or path.stat().st_size != metadata.size:
        return False
    if metadata.lfs:
        return file_digest(path) == metadata.lfs.sha256
    return git_blob_digest(path) == metadata.blob_id


def prepare_template(base, repo, revision, cache, log=print):
    from huggingface_hub import HfApi
    from huggingface_hub.utils import validate_repo_id
    validate_repo_id(repo)
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Pin the SGLang template to a full commit SHA")
    info = HfApi(token=True).model_info(repo, revision=revision, files_metadata=True)
    if info.sha != revision:
        raise ValueError("Hub returned a different template revision")
    inventory = {f.rfilename: f for f in info.siblings}
    small = ["config.json", "README.md", *TOKENIZER_FILES]
    if any(name not in inventory for name in [*small, *WEIGHTS]):
        raise ValueError("The template must contain the complete CKermit Base package")
    verified = {}
    for name, relative in WEIGHTS.items():
        log(f"[release] Verifying local {relative} against {repo}@{revision[:12]}")
        if not inventory[name].lfs or not matches_hub_file(Path(base) / relative, inventory[name]):
            raise ValueError(f"Template/base weights differ: {name}; do not merge this adapter into another base")
        verified[name] = {"size": inventory[name].size, "sha256": inventory[name].lfs.sha256}
    # These big files are already on disk and verified against the remote commit.
    # Download only packaging/tokenizer metadata, never the duplicate 12 GB weights.
    cache = Path(cache).resolve() / repo.replace("/", "--") / revision
    download_files(cache, small, repo, revision, repo_type="model", authenticated=True, log=log)
    for name in small:
        if not matches_hub_file(cache / name, inventory[name]):
            raise ValueError(f"Template file checksum mismatch: {name}")
    return cache, verified


def verify_tokenizer(base, template):
    from .model_manager import ModelManager
    ModelManager._ensure_upstream_namespace()
    from fireredtts3.utils.text_tokenizer import load_text_tokenizer
    from transformers import AutoTokenizer
    original = load_text_tokenizer(str(Path(base) / "text_tokenizer"))
    packaged = AutoTokenizer.from_pretrained(str(template), local_files_only=True, trust_remote_code=False)
    # The original loader adds FireRed's language/control tokens at runtime.
    # CKermit's package already contains them. Compare the effective tokenizer.
    if (original.get_vocab() != packaged.get_vocab()
            or json.loads(original.backend_tokenizer.to_str()) != json.loads(packaged.backend_tokenizer.to_str())):
        raise ValueError("Packaged tokenizer differs from the effective training tokenizer")
    probes = ["<|Russian|><|sot|>Вы уже отправили файл? Да, готово.<|eot|>",
              "Ёжик съел яблоко. Всё хорошо?", "Hello, what time is it?", "你好，你好吗？"]
    if any(original.encode(t, add_special_tokens=False) != packaged.encode(t, add_special_tokens=False) for t in probes):
        raise ValueError("Tokenizer encoding parity failed")
    return {"vocabulary_size": len(original), "backend_equal": True, "encoding_probes": len(probes)}


def validate_template_config(config):
    if (config.get("model_type") != "fireredtts3"
            or config.get("architectures") != ["FireRedTTS3ForConditionalGeneration"]
            or config.get("use_lm_head") is not False or config.get("speaker_token") is not None
            or config.get("text_latent_interleave_ratio") is not None or config.get("redae_scale") != 1.0
            or config.get("dit_in_channels") != sum(config.get(k, 0) for k in ("redae_dim", "spk_in_dim", "dit_hidden_size"))):
        raise ValueError("Expected the non-interleaved FireRedTTS3 Base SGLang Omni configuration")


def model_card(provenance, evaluation=None):
    source = provenance["template"]
    text = f"""---
license: apache-2.0
language:
  - ru
pipeline_tag: text-to-speech
base_model:
  - {source['repo']}
  - FireRedTeam/FireRedTTS3
tags:
  - sglang-omni
  - fireredtts3
  - voice-cloning
  - lora-merged
---

# FireRedTTS3-Base — Russian adaptation

This is a full FP32 checkpoint with the supplied Russian LoRA merged into the
FireRedTTS3 Base core. No separate adapter is needed for serving. The package
includes the tokenizer, RedAE and CAM++ assets required for zero-shot cloning.

The packaging template is [{source['repo']}](https://huggingface.co/{source['repo']})
at commit `{source['revision']}`. Its core/RedAE/CAM++ files were verified to be
identical to the original local base used by the adapter. The training base is
bound by its complete artifact fingerprint, recorded in `export_provenance.json`.

Merge validation checked PEFT layer outputs before/after merge, unchanged frozen
tensors, every saved core tensor, and the effective tokenizer. This is **not** an
end-to-end SGLang GPU parity test. Runtime precision and generation settings can
change the output compared with the training inference pipeline.

## SGLang Omni

Use SGLang **Omni** with FireRedTTS3 support. The inspected integration is
[PR #3005]({OMNI_PR}), revision `{OMNI_REVISION}`; an older SGLang/Omni installation
does not provide this model. Follow its installation prerequisites in a separate
environment from training. Download the entire model repository.

From the compatible sglang-omni checkout, serve the downloaded directory:

```bash
CUDA_VISIBLE_DEVICES=0 sgl-omni serve \\
  --model-path /absolute/path/to/downloaded-model \\
  --config examples/configs/firered_tts3_base.yaml \\
  --allowed-local-media-path /absolute/path/to/reference-audio \\
  --port 8000
```

Wait for `/health` to return HTTP 200. Call `/v1/audio/speech` with `language`
set to `Russian`, a target `input`, and one `references` entry containing
`audio_path` and the reference recording's actual `text`. Output is 24 kHz mono.
The inspected API limits `input` to 400 characters. See `UPSTREAM_MODEL_CARD.md`
and the integration cookbook for streaming and generation settings.

## Provenance and license

Original model: [FireRedTeam/FireRedTTS3](https://huggingface.co/FireRedTeam/FireRedTTS3).
Packaging: CKermit / SGLang Omni. The original Apache-2.0 license is included in
`LICENSE`; upstream attribution and usage guidance are retained in
`UPSTREAM_MODEL_CARD.md`. This package contains modified model weights.
The LoRA profile is `{provenance['profile']}`. Training audio, reference recordings,
optimizer state and authentication credentials are not part of this release.
"""
    if evaluation is not None:
        text += f"""
## Reported evaluation before merging

The supplied standalone-pipeline comparison contains {evaluation['cases']} cases
and {evaluation['speakers']} speakers. Mean speaker-similarity delta:
`{evaluation['mean_similarity_delta']:.8f}`; worst per-speaker mean delta:
`{evaluation['worst_speaker_mean_delta']:.8f}`; mean WER delta:
`{evaluation['mean_wer_delta']:.8f}`. These are absolute metric differences against
the supplied baseline, not percentages of perceptual quality. A small benchmark
does not establish general voice-cloning or prosody parity. These scores were
not measured in SGLang or on the merged checkpoint. Report hashes and scorer IDs
are recorded in `export_provenance.json`; no reference audio/text is published.
"""
    return text


def seal_release(root):
    root = Path(root)
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("Release files must be independent files, not symlinks")
        if path.is_file() and path.name != MANIFEST:
            files[path.relative_to(root).as_posix()] = {"size": path.stat().st_size, "sha256": file_digest(path)}
    identity = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    manifest = {"schema": 1, "producer": PRODUCER, "release_id": identity, "files": files}
    write_json(root / MANIFEST, manifest)
    return manifest


def verify_release(root):
    root = Path(root).resolve()
    manifest = json.loads((root / MANIFEST).read_text())
    if manifest.get("schema") != 1 or manifest.get("producer") != PRODUCER:
        raise ValueError("Not a verified FireRedTTS3 release")
    files = manifest["files"]
    identity = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if manifest.get("release_id") != identity:
        raise ValueError("Release manifest identity mismatch")
    actual = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("Symlinks are not allowed in a published release")
        if path.is_file():
            actual.add(path.relative_to(root).as_posix())
    if actual != set(files) | {MANIFEST}:
        raise ValueError("Release contains missing/extra files; do not put tokens, audio or training data here")
    for name, record in files.items():
        path = root / name
        if (Path(name).is_absolute() or ".." in Path(name).parts or "\\" in name
                or path.stat().st_size != record["size"] or file_digest(path) != record["sha256"]):
            raise ValueError(f"Release checksum mismatch: {name}")
    required = {"README.md", "LICENSE", "config.json", "tokenizer.json", "export_provenance.json",
                "redae/model.safetensors", "campp/campplus_voxceleb.bin"}
    if not required <= set(files) or not any("/" not in n and n.endswith(".safetensors") for n in files):
        raise ValueError("Incomplete serving package")
    return manifest


def build_release(base, adapter, output, *, template_repo=TEMPLATE_REPO, template_revision=TEMPLATE_REVISION,
                  template_cache=".runtime/export-templates", evaluation_baseline=None, evaluation_adapted=None, log=print):
    from filelock import FileLock
    from .evaluation import compare_reports
    base, adapter, output = Path(base).resolve(), Path(adapter).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError("Export output must not exist")
    if bool(evaluation_baseline) != bool(evaluation_adapted):
        raise ValueError("Supply both baseline and adapted evaluation reports, or neither")
    # Check the local adapter before any Hub requests or large base hashing.
    for name in ("adapter_config.json", "adapter_model.safetensors", "firered_lora_metadata.json"):
        if not (adapter / name).is_file():
            raise ValueError(f"Missing adapter artifact: {name}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output.parent / f".{output.name}.release.lock"), timeout=0):
        if output.exists():
            raise ValueError("Export output must not exist")
        template, verified = prepare_template(base, template_repo, template_revision, template_cache, log)
        validate_template_config(json.loads((template / "config.json").read_text()))
        tokenizer = verify_tokenizer(base, template)
        evaluation = None
        if evaluation_baseline:
            adapted = json.loads(Path(evaluation_adapted).read_text())
            if not adapted.get("adapter") or Path(adapted["adapter"]).resolve() != adapter:
                raise ValueError("Evaluation report names a different adapter")
            evaluation = compare_reports(evaluation_baseline, evaluation_adapted)
        with tempfile.TemporaryDirectory(prefix=".release-", dir=output.parent) as temporary:
            staging = Path(temporary) / "package"
            export_merged(base, adapter, template / "config.json", staging, tokenizer_dir=template, log=log)
            for name in ("redae/model.safetensors", "campp/campplus_voxceleb.bin"):
                if file_digest(staging / name) != verified[name]["sha256"]:
                    raise ValueError(f"Copied acoustic asset differs from the verified template: {name}")
            for name in TOKENIZER_FILES:
                if file_digest(staging / name) != file_digest(template / name):
                    raise ValueError(f"Copied tokenizer asset changed: {name}")
            provenance = json.loads((staging / "export_provenance.json").read_text())
            # Local paths and the full training recipe must not become public metadata.
            provenance.pop("base", None)
            provenance.pop("adapter", None)
            provenance.update(schema=2, created_at=datetime.now(timezone.utc).isoformat(),
                              template={"repo": template_repo, "revision": template_revision},
                              template_weight_files_verified=verified, tokenizer=tokenizer,
                              sglang_integration={"pull_request": OMNI_PR, "revision": OMNI_REVISION})
            if evaluation is not None:
                provenance["evaluation"] = {"comparison": evaluation,
                    "baseline_report_sha256": file_digest(evaluation_baseline),
                    "adapted_report_sha256": file_digest(evaluation_adapted),
                    "speaker_model": adapted["speaker_model"], "asr_model": adapted["asr_model"],
                    "scope": "Unmerged adapter in the standalone pipeline; not SGLang runtime"}
            write_json(staging / "export_provenance.json", provenance)
            shutil.copy2(template / "README.md", staging / "UPSTREAM_MODEL_CARD.md")
            shutil.copy2(Path(__file__).resolve().parents[1] / "LICENSE_UPSTREAM", staging / "LICENSE")
            (staging / "README.md").write_text(model_card(provenance, evaluation), encoding="utf-8")
            manifest = seal_release(staging)
            if output.exists():
                raise ValueError("Export output was created by another process")
            staging.rename(output)
    result = {"output": str(output), "release_id": manifest["release_id"], "files": len(manifest["files"]) + 1,
              "bytes": sum(f["size"] for f in manifest["files"].values()), "sglang_runtime_validated": False}
    log("[release] Complete; model package is ready for publication")
    return result


def remote_matches(info, root, names):
    inventory = {f.rfilename: f for f in info.siblings}
    if not names <= set(inventory) or set(inventory) - names - {".gitattributes"}:
        return False
    return all(matches_hub_file(root / name, inventory[name]) for name in sorted(names))


def publish_release(output, repo_name="FireRedTTS3-Base-Russian", *, public=False, log=print):
    from huggingface_hub import HfApi
    from huggingface_hub.errors import RepositoryNotFoundError
    from huggingface_hub.utils import validate_repo_id
    root = Path(output).resolve()
    manifest = verify_release(root)
    if "/" in repo_name:
        raise ValueError("Use a repository NAME only; the account comes from the local HF token")
    api = HfApi(token=True)
    owner = api.whoami()["name"]
    repo_id = f"{owner}/{repo_name}"
    validate_repo_id(repo_id)
    names = set(manifest["files"]) | {MANIFEST}
    log(f"[publish] Account: {owner}; repository: {repo_id}; visibility: {'public' if public else 'private'}")
    try:
        info = api.model_info(repo_id, files_metadata=True)
    except RepositoryNotFoundError:
        api.create_repo(repo_id, repo_type="model", private=not public, exist_ok=False)
        info = api.model_info(repo_id, files_metadata=True)
    if info.private != (not public):
        raise ValueError("Existing repository visibility differs; refusing to change it automatically")
    existing = {f.rfilename for f in info.siblings} - {".gitattributes"}
    if existing:
        if not remote_matches(info, root, names):
            raise ValueError("Destination already contains different files; use a new repository name")
        commit = info.sha
        log("[publish] Identical release already exists; no files uploaded")
    else:
        # Exact allowlist + an optimistic parent bind. No recursive upload of a
        # training directory and no deletion/overwrite of an unrelated model.
        result = api.upload_folder(repo_id=repo_id, repo_type="model", folder_path=str(root),
            allow_patterns=sorted(names), parent_commit=info.sha,
            commit_message=f"Publish merged FireRedTTS3 release {manifest['release_id'][:12]}")
        commit = result.oid
        info = api.model_info(repo_id, revision=commit, files_metadata=True)
        if not remote_matches(info, root, names):
            raise ValueError("Hub commit exists but remote content verification failed; inspect before serving")
    result = {"repo_id": repo_id, "url": f"https://huggingface.co/{repo_id}", "commit": commit,
              "revision_url": f"https://huggingface.co/{repo_id}/tree/{commit}", "public": public,
              "release_id": manifest["release_id"], "remote_files_verified": True}
    write_json(root.parent / f"{root.name}.published.json", result)
    return result

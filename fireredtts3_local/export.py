"""Merge all trained core components, preserving the SGLang Omni key layout."""
from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
import shutil
import tempfile

import torch

from .training_data import file_digest, model_identity

TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                   "added_tokens.json", "vocab.json", "merges.txt", "chat_template.jinja")


def tensor_digest(tensor):
    """Hash/validate large CPU tensors in bounded blocks, including BF16."""
    digest = hashlib.sha256()
    flat = tensor.detach().reshape(-1)
    for start in range(0, flat.numel(), 1024 * 1024):
        block = flat[start:start + 1024 * 1024].cpu().contiguous()
        if block.is_floating_point() and not torch.isfinite(block).all():
            raise ValueError("Non-finite checkpoint tensor")
        digest.update(memoryview(block.view(torch.uint8).numpy()))
    return digest.hexdigest()


def validate_export_config(core, template):
    if template.get("model_type") != "fireredtts3":
        raise ValueError("Expected a FireRedTTS3 SGLang configuration")
    for field in ("redae_dim", "num_history_patches", "patch_size", "patch_encoder_hidden_size",
                  "patch_encoder_mlp_ratio", "patch_encoder_depth", "patch_encoder_num_heads",
                  "dit_mlp_ratio", "dit_depth", "dit_num_heads", "dit_hidden_size", "spk_in_dim"):
        if template.get(field) != getattr(core.config, field):
            raise ValueError(f"SGLang/core config mismatch: {field}")
    for field in ("hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads",
                  "num_key_value_heads", "vocab_size", "head_dim", "hidden_act", "rms_norm_eps",
                  "max_position_embeddings", "tie_word_embeddings", "attention_bias",
                  "bos_token_id", "eos_token_id"):
        if template.get(field) != getattr(core.backbone_llm.config, field):
            raise ValueError(f"SGLang/backbone config mismatch: {field}")
    backbone = core.backbone_llm.config
    rope_theta = getattr(backbone, "rope_theta", None)
    if rope_theta is None:
        rope_theta = (getattr(backbone, "rope_parameters", None) or {}).get("rope_theta")
    if template.get("rope_theta") != rope_theta:
        raise ValueError("SGLang/backbone config mismatch: rope_theta")


def export_merged(base, adapter, sglang_config, output, *, tokenizer_dir=None, log=print):
    from peft import PeftModel
    from safetensors import safe_open
    from .model_manager import ModelManager
    from .training_objective import adapter_targets, validate_adapter_targets
    from .training_run import initial_adapter_info
    ModelManager._ensure_upstream_namespace()
    from fireredtts3.llm.fireredtts3_base import FireRedTTS3BaseCore
    from fireredtts3.redae.redae import RedAEConfig
    base, adapter, output = Path(base).resolve(), Path(adapter).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError("Export output must not exist")
    for folder in ("fireredtts3_base", "text_tokenizer", "redae", "campp"):
        if not (base/folder).is_dir():
            raise ValueError(f"Missing original model component: {folder}")
    template = json.loads(Path(sglang_config).read_text())
    metadata = json.loads((adapter / "firered_lora_metadata.json").read_text())
    config = json.loads((adapter / "adapter_config.json").read_text())
    profile = metadata.get("recipe", {}).get("options", {}).get("profile")
    initial = initial_adapter_info(adapter, config["r"], config["lora_alpha"], config["lora_dropout"], profile)
    log("[export] Checking original base/tokenizer fingerprint")
    if metadata.get("base_fingerprint") != model_identity(base):
        raise ValueError("Adapter was trained against different base artifacts/tokenizer")
    redae = RedAEConfig.from_pretrained(str(base / "redae"))
    for field, value in vars(redae).items():
        if field.startswith(("enc_", "dec_")) or field in {"audio_patch_size", "audio_sample_rate", "bottleneck_dim"}:
            if template.get(field) != value:
                raise ValueError(f"SGLang/RedAE config mismatch: {field}")
    # CPU export avoids loading the full synthesis pipeline onto a GPU.
    previous_backend = os.environ.get("FIRERED_ATTN_IMPLEMENTATION")
    try:
        os.environ["FIRERED_ATTN_IMPLEMENTATION"] = "sdpa"
        core = FireRedTTS3BaseCore.from_pretrained(str(base/"fireredtts3_base"), dtype=torch.float32)
    finally:
        if previous_backend is None:
            os.environ.pop("FIRERED_ATTN_IMPLEMENTATION", None)
        else:
            os.environ["FIRERED_ATTN_IMPLEMENTATION"] = previous_backend
    validate_export_config(core, template)
    if core.config.redae_dim != redae.bottleneck_dim:
        raise ValueError("Core/RedAE latent dimensions differ")
    targets = adapter_targets(core, profile)
    validate_adapter_targets(core, initial["targets"], targets)
    expected = {k: tuple(v.shape) for k, v in core.state_dict().items()}
    original = {k: tensor_digest(v) for k, v in core.state_dict().items()}
    model = PeftModel.from_pretrained(core, str(adapter))
    model.eval()
    # Real PEFT forwards before merging, not just a second implementation of B@A.
    generator = torch.Generator(device="cpu").manual_seed(1234)
    probes = {}
    with torch.inference_mode():
        for name in targets:
            layer = model.get_base_model().get_submodule(name)
            x = torch.randn(2, layer.in_features, generator=generator) * .1
            probes[name] = (x, layer(x).clone())
    log(f"[export] Merging {len(targets)} LoRA modules in FP32 on CPU")
    merged = model.merge_and_unload(safe_merge=True)
    actual = {k: tuple(v.shape) for k, v in merged.state_dict().items()}
    if actual != expected:
        raise ValueError("Merged core keys/shapes differ from the original core")
    errors = []
    with torch.inference_mode():
        for name, (x, before) in probes.items():
            after = merged.get_submodule(name)(x)
            torch.testing.assert_close(after, before, rtol=1e-4, atol=1e-5)
            errors.append(float((after - before).abs().max()))
    final = {k: tensor_digest(v) for k, v in merged.state_dict().items()}
    changed = {key for key in original if original[key] != final[key]}
    if changed - {name + ".weight" for name in targets}:
        raise ValueError("Merge changed weights outside the declared LoRA target modules")
    provenance = {"base": str(base.resolve()), "adapter": str(adapter.resolve()),
                  "sglang_config_sha256": file_digest(sglang_config), "base_fingerprint": metadata["base_fingerprint"],
                  "merged_tensor_count": len(actual),
                  "adapter_fingerprint": initial["fingerprint"], "profile": profile,
                  "changed_tensors": len(changed), "layer_probes": len(probes),
                  "layer_probe_max_abs_error": max(errors, default=0.0),
                  "unchanged_tensors_verified": len(original) - len(changed),
                  "merge_dtype": "float32", "saved_tensor_hashes_verified": True,
                  "sglang_runtime_validated": False,
                  "note": "FP32 PEFT layer probes passed. End-to-end GPU/SGLang synthesis parity is not asserted."}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".merge-", dir=output.parent) as temporary:
        staging = Path(temporary) / "model"
        merged.save_pretrained(staging, safe_serialization=True, max_shard_size="4GB")
        saved = {}
        for shard in staging.glob("*.safetensors"):
            with safe_open(shard, framework="pt", device="cpu") as tensors:
                for key in tensors.keys():
                    if key in saved or tensor_digest(tensors.get_tensor(key)) != final.get(key):
                        raise ValueError(f"Saved tensor differs from merged core: {key}")
                    saved[key] = tuple(tensors.get_slice(key).get_shape())
        if saved != expected:
            raise ValueError("Saved checkpoint keys/shapes do not cover the complete core")
        (staging/"config.json").write_text(json.dumps(template, indent=2), encoding="utf-8")
        tokenizer = Path(tokenizer_dir) if tokenizer_dir else base / "text_tokenizer"
        for name in TOKENIZER_FILES:
            if (tokenizer/name).is_file():
                shutil.copy2(tokenizer/name, staging/name)
        for folder in ("redae", "campp"):
            (staging/folder).mkdir()
            for file in (base/folder).iterdir():
                if file.is_file() and (file.name in {"config.json", "model.safetensors", "model.safetensors.index.json",
                                                     "campplus_voxceleb.bin"}
                                       or file.name.startswith("model-") and file.suffix == ".safetensors"):
                    shutil.copy2(file, staging/folder/file.name)
        (staging/"export_provenance.json").write_text(json.dumps(provenance, indent=2))
        if output.exists():
            raise ValueError("Export output was created by another process")
        staging.rename(output)
    log(f"[export] Verified {len(saved)} saved tensors; layer max error={max(errors, default=0.):.3g}")
    return output

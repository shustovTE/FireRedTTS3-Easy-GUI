"""Merge all trained core components, preserving the SGLang Omni key layout."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil

from .training_data import file_digest, model_identity


def validate_export_config(core, template):
    if template.get("model_type") != "fireredtts3":
        raise ValueError("Expected a FireRedTTS3 SGLang configuration")
    for field in ("redae_dim", "num_history_patches", "patch_size", "patch_encoder_hidden_size",
                  "patch_encoder_mlp_ratio", "patch_encoder_depth", "patch_encoder_num_heads",
                  "dit_mlp_ratio", "dit_depth", "dit_num_heads", "dit_hidden_size", "spk_in_dim"):
        if template.get(field) != getattr(core.config, field):
            raise ValueError(f"SGLang/core config mismatch: {field}")
    for field in ("hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads",
                  "num_key_value_heads", "vocab_size", "head_dim"):
        if template.get(field) != getattr(core.backbone_llm.config, field):
            raise ValueError(f"SGLang/backbone config mismatch: {field}")
    backbone = core.backbone_llm.config
    rope_theta = getattr(backbone, "rope_theta", None)
    if rope_theta is None:
        rope_theta = (getattr(backbone, "rope_parameters", None) or {}).get("rope_theta")
    if template.get("rope_theta") != rope_theta:
        raise ValueError("SGLang/backbone config mismatch: rope_theta")


def export_merged(base, adapter, sglang_config, output):
    from peft import PeftModel
    from .model_manager import ModelManager
    ModelManager._ensure_upstream_namespace()
    from fireredtts3.llm.fireredtts3_base import FireRedTTS3BaseCore
    from fireredtts3.redae.redae import RedAEConfig
    base, adapter, output = Path(base), Path(adapter), Path(output)
    if output.exists():
        raise ValueError("Export output must not exist")
    for folder in ("fireredtts3_base", "text_tokenizer", "redae", "campp"):
        if not (base/folder).is_dir():
            raise ValueError(f"Missing original model component: {folder}")
    template = json.loads(Path(sglang_config).read_text())
    metadata = json.loads((adapter / "firered_lora_metadata.json").read_text())
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
        core = FireRedTTS3BaseCore.from_pretrained(str(base/"fireredtts3_base"))
    finally:
        if previous_backend is None:
            os.environ.pop("FIRERED_ATTN_IMPLEMENTATION", None)
        else:
            os.environ["FIRERED_ATTN_IMPLEMENTATION"] = previous_backend
    validate_export_config(core, template)
    if core.config.redae_dim != redae.bottleneck_dim:
        raise ValueError("Core/RedAE latent dimensions differ")
    expected = {k: tuple(v.shape) for k, v in core.state_dict().items()}
    model = PeftModel.from_pretrained(core, str(adapter))
    merged = model.merge_and_unload(safe_merge=True)
    actual = {k: tuple(v.shape) for k, v in merged.state_dict().items()}
    if actual != expected:
        raise ValueError("Merged core keys/shapes differ from the original core")
    output.mkdir(parents=True)
    merged.save_pretrained(output, safe_serialization=True)
    from safetensors import safe_open
    saved = {}
    for shard in output.glob("*.safetensors"):
        with safe_open(shard, framework="pt", device="cpu") as tensors:
            for key in tensors.keys():
                saved[key] = tuple(tensors.get_slice(key).get_shape())
    if saved != expected:
        raise ValueError("Saved checkpoint keys/shapes do not cover the complete core")
    (output/"config.json").write_text(json.dumps(template, indent=2), encoding="utf-8")
    for file in (base/"text_tokenizer").iterdir():
        if file.is_file() and (file.name.startswith(("tokenizer", "special_tokens", "added_tokens", "vocab", "merges", "chat_template"))):
            shutil.copy2(file, output/file.name)
    for folder in ("redae", "campp"):
        shutil.copytree(base/folder, output/folder)
    provenance = {"base": str(base.resolve()), "adapter": str(adapter.resolve()),
                  "sglang_config_sha256": file_digest(sglang_config), "base_fingerprint": metadata["base_fingerprint"],
                  "merged_tensor_count": len(actual),
                  "sglang_runtime_validated": False,
                  "note": "Full core merged; GPU synthesis parity and SGLang runtime checks are still required."}
    (output/"export_provenance.json").write_text(json.dumps(provenance, indent=2))
    return output

"""Publish the approved step-500 backbone listening variant to the existing private HF model."""
import json
from pathlib import Path

from .paths import MODEL_DIR, ROOT
from .release import build_release, publish_release
from .training_data import file_digest
from .training_run import initial_adapter_info
from .vox_recipe_transfer import BASE_SHA
from .vox_recovery import FOUR_PATCH

REPO = "shustov-te/FireRedTTS3-Base-Russian"
RUN = ROOT / "training/outputs/firered-vox-recipe-12-cap30m-levels/lora-base-recovery-4patch-0500"
OUTPUT = ROOT / "training/exports/FireRedTTS3-Base-Russian-4patch-0500-backbone"


def publish(run=RUN, output=OUTPUT):
    from huggingface_hub import HfApi

    run, output = Path(run).resolve(), Path(output).resolve()
    adapter = run / "lora/step-0500"
    report = json.loads((run / "audio/step-0500/series-2/backbone/report.json").read_text())
    plan = report["plan"]
    checkpoint = plan["checkpoints"]["step-0500"]
    files = ("checkpoint.json", "adapter_config.json", "adapter_model.safetensors")
    if (plan["variants"] != ["backbone"]
            or plan["base_sha256"] != BASE_SHA or Path(checkpoint["path"]).resolve() != adapter
            or checkpoint["hashes"] != {name: file_digest(adapter/name) for name in files}):
        raise ValueError("The checkpoint differs from the backbone listening run")
    record = json.loads((adapter/"checkpoint.json").read_text())
    recipe = record["recipe"]
    if (record["step"] != 500 or recipe["mode"] != "lora" or recipe["base_sha256"] != BASE_SHA
            or recipe.get("acoustic_recovery") != FOUR_PATCH or recipe.get("rise_gate")):
        raise ValueError("Expected the ungated four-patch LoRA at step 500")
    config = json.loads((adapter/"adapter_config.json").read_text())
    initial = initial_adapter_info(adapter, config["r"], config["lora_alpha"], config["lora_dropout"], "vox_attention")
    remote = HfApi(token=True).model_info(REPO, revision="main")
    if remote.private is not True:
        raise ValueError(f"Destination must be private: {REPO}")
    if not output.exists():
        build_release(MODEL_DIR, adapter, output, backbone_only=True)
    provenance = json.loads((output/"export_provenance.json").read_text())
    expected = dict(merged_variant="backbone", checkpoint_step=500, profile="vox_attention",
                    base_fingerprint=BASE_SHA, adapter_fingerprint=initial["fingerprint"])
    if any(provenance.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Export does not match the selected backbone checkpoint: {output}")
    return publish_release(output, repo_id=REPO, public=False, update_existing=True, expected_parent=remote.sha)


if __name__ == "__main__":
    import argparse
    argparse.ArgumentParser(description=__doc__).parse_args()
    print(json.dumps(publish(), ensure_ascii=False, indent=2))

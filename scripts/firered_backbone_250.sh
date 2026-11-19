#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")/.."

root=training/outputs/firered-vox-recipe-12-cap30m-levels
run="$root/lora-base-recovery-4patch-0250"
python=.venv/bin/python

"$python" -u -m fireredtts3_local.vox_recipe_sweep --multistep-from-base --until 250 \
  "$root/comparison-0030-yrslvj2s/report.json" \
  "$root/comparison-0030-ql21g14n/report.json"

"$python" -u -m fireredtts3_local.vox_attention_ablation \
  --full-backbone "$run/audio/step-0250"

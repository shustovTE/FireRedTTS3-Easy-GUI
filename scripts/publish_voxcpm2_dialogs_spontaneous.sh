#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

python=.venv-voxcpm2/bin/python
run=voxcpm2-dialogs-spontaneous-full-1000
parent=$("$python" -c 'from huggingface_hub import HfApi
from fireredtts3_local.voxcpm_release import DESTINATION
print(HfApi(token=True).model_info(DESTINATION, revision="main").sha)')
printf 'Current Hugging Face parent: %s\n' "$parent"

exec "$python" -m fireredtts3_local.voxcpm_release \
  --full-checkpoint "training/outputs/$run/step_0001000" \
  --full-recipe "training/jobs/$run/recipe.yaml" \
  --output "training/exports/$run" \
  --expected-parent "$parent" --publish

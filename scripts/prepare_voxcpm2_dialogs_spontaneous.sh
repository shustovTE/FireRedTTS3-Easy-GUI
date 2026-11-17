#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
python=.venv-voxcpm2/bin/python
recipe=training/recipes/voxcpm2-dialogs-spontaneous-full-1000.yaml

uv pip install --python "$python" "dpdfnet==0.6.0"
"$python" -m fireredtts3_local.voxcpm_validated_sources --spontaneous
if [[ -e "$recipe" ]]; then
    "$python" -m fireredtts3_local.voxcpm_training audit "$recipe"
else
    "$python" -m fireredtts3_local.voxcpm_training build-dialogs-spontaneous
fi
printf '[prepare] COMPLETE %s\n' "$recipe"

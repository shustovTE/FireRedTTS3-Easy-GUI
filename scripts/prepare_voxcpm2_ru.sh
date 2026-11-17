#!/usr/bin/env bash
# Run from this repository, or launch with nohup as documented.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
python_path=".venv-voxcpm2/bin/python"
test -x "$python_path" || { echo 'Run voxcpm_training setup first.' >&2; exit 1; }
mkdir -p .runtime training/logs
exec 9>.runtime/voxcpm-preparation.lock
flock -n 9 || { echo 'Another VoxCPM preparation is running.' >&2; exit 1; }
status_file="training/logs/voxcpm2-prepare.status"
printf 'running\n' > "$status_file"
trap 'code=$?; if [ "$code" -ne 0 ]; then printf "failed (exit %s)\n" "$code" > "$status_file"; fi' EXIT

if [ -e training/datasets/cv-scripted-27 ]; then
    "$python_path" -m fireredtts3_local.common_voice_scripted audit \
        --output training/datasets/cv-scripted-27
else
    "$python_path" -m fireredtts3_local.common_voice_scripted prepare \
        --output training/datasets/cv-scripted-27 \
        --train-minutes 30 --validation-seconds 200 --seed 1234 \
        --anchor-train \
            training/datasets/dialogs-base-08/train.jsonl \
            training/datasets/cv-spontaneous-09/train.jsonl \
        --anchor-validation \
            training/datasets/dialogs-base-08/validation.jsonl \
            training/datasets/cv-spontaneous-09/validation.jsonl
fi

"$python_path" -m fireredtts3_local.voxcpm_data prepare

if [ -e training/recipes/voxcpm2-ru.yaml ]; then
    "$python_path" -m fireredtts3_local.voxcpm_training audit training/recipes/voxcpm2-ru.yaml
else
    "$python_path" -m fireredtts3_local.voxcpm_training build-recipe \
        --dataset training/datasets/voxcpm2-ru \
        --output training/recipes/voxcpm2-ru.yaml \
        --output-dir training/outputs/voxcpm2-ru-lora \
        --epochs 1 --batch-size 1 --grad-accum-steps 8
fi
printf 'completed\n' > "$status_file"
printf '[prepare] COMPLETE: training/recipes/voxcpm2-ru.yaml\n'

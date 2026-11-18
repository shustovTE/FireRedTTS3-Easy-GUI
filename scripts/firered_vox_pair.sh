#!/usr/bin/env bash
# Train both independent FireRed candidates up to a matched checkpoint, then compare.
set -euo pipefail
cd -- "$(dirname -- "$0")/.."
if [[ $# -lt 1 || $# -gt 2 || ! $1 =~ ^(30|500|1000)$ ]]; then
    echo "Usage: bash scripts/firered_vox_pair.sh {30|500|1000} [benchmark.jsonl]" >&2
    exit 2
fi
step=$1
python=.venv/bin/python
module=fireredtts3_local.vox_recipe_transfer
output=training/outputs/firered-vox-recipe-12
printf -v checkpoint 'step-%04d' "$step"
shopt -s nullglob
for mode in lora full; do
    if [[ -f "$output/$mode/$checkpoint/checkpoint.json" ]]; then
        continue
    fi
    saved=("$output/$mode"/step-*/checkpoint.json)
    resume=()
    if (( ${#saved[@]} )); then
        resume=(--resume)
    fi
    "$python" -u -m "$module" train "$mode" --until "$step" "${resume[@]}"
done
benchmark=()
if (( $# == 2 )); then
    benchmark=(--benchmark "$2")
fi
"$python" -u -m "$module" compare --step "$step" "${benchmark[@]}"

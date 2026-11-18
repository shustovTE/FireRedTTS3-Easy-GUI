#!/usr/bin/env bash
# Train both independent FireRed candidates up to a matched checkpoint, then compare.
set -euo pipefail
cd -- "$(dirname -- "$0")/.."
levels=0
if [[ ${1:-} == --levels ]]; then
    levels=1
    shift
fi
if [[ $# -eq 0 || ! $1 =~ ^(30|500|1000)$ ]] || (( !levels && $# > 2 )); then
    echo "Usage: $0 {30|500|1000} [benchmark.jsonl] | --levels {30|500|1000} [original-report.json ...]" >&2
    exit 2
fi
step=$1
shift
python=.venv/bin/python
module=fireredtts3_local.vox_recipe_transfer
output=training/outputs/firered-vox-recipe-12-cap30m
data=()
if (( levels )); then
    for report in "$@"; do
        if [[ ! -f "$report" || ! -f "$(dirname -- "$report")/benchmark.jsonl" ]]; then
            echo "Missing original report or its benchmark.jsonl: $report" >&2
            exit 2
        fi
    done
    dataset=training/datasets/firered-vox-cap30m-levels
    output=training/outputs/firered-vox-recipe-12-cap30m-levels
    "$python" -u -m fireredtts3_local.vox_recipe_levels --output "$dataset"
    data=(--dataset "$dataset")
fi
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
    "$python" -u -m "$module" train "$mode" --output "$output" --until "$step" "${data[@]}" "${resume[@]}"
done
compare=("$python" -u -m "$module" compare --output "$output" --step "$step" "${data[@]}")
if (( $# == 0 )); then
    "${compare[@]}"
elif (( levels )); then
    for report in "$@"; do
        "${compare[@]}" --against "$report"
    done
else
    "${compare[@]}" --benchmark "$1"
fi

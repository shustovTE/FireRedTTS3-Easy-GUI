#!/usr/bin/env bash
# Train the selected FireRed candidates up to a checkpoint, then replay the benchmarks.
set -euo pipefail
cd -- "$(dirname -- "$0")/.."
levels=0
modes=(lora full)
operation=${1:-}
case "$operation" in
    --levels) levels=1; shift ;;
    --resume-levelled-lora) levels=1; modes=(lora); shift ;;
esac
if [[ $# -eq 0 || ! $1 =~ ^(30|500|1000)$ ]] || (( !levels && $# > 2 )); then
    echo "Usage: $0 {30|500|1000} [benchmark.jsonl] | {--levels|--resume-levelled-lora} {30|500|1000} [report.json ...]" >&2
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
    if [[ $operation == --levels ]]; then
        "$python" -u -m fireredtts3_local.vox_recipe_levels --output "$dataset"
    elif [[ ! -f "$dataset/level_report.json" ]]; then
        echo "Missing the levelled dataset used by the LoRA pilot: $dataset" >&2
        exit 2
    fi
    data=(--dataset "$dataset")
fi
printf -v checkpoint 'step-%04d' "$step"
shopt -s nullglob
for mode in "${modes[@]}"; do
    if [[ -f "$output/$mode/$checkpoint/checkpoint.json" ]]; then
        continue
    fi
    saved=("$output/$mode"/step-*/checkpoint.json)
    resume=()
    if (( ${#saved[@]} )); then
        resume=(--resume)
    elif [[ $operation == --resume-levelled-lora ]]; then
        echo "No saved LoRA checkpoint to resume in $output/$mode" >&2
        exit 2
    fi
    "$python" -u -m "$module" train "$mode" --output "$output" --until "$step" "${data[@]}" "${resume[@]}"
done
compare=("$python" -u -m "$module" compare --output "$output" --step "$step" "${data[@]}")
if [[ $operation == --resume-levelled-lora ]]; then
    compare+=(--lora-only)
fi
if (( $# == 0 )); then
    "${compare[@]}"
elif (( levels )); then
    for report in "$@"; do
        "${compare[@]}" --against "$report"
    done
else
    "${compare[@]}" --benchmark "$1"
fi

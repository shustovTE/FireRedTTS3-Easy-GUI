#!/usr/bin/env bash
# Stop the gated resonant run after its first on-disk checkpoint.
set -eu
cd -- "$(dirname -- "$0")/.."
run=training/outputs/firered-vox-recipe-12-cap30m-levels/lora-base-recovery-4patch-0500-rise20pct-resonant-w015
checkpoint="$run/lora/step-0100"
lock="$run/.run.lock"
if [[ ! -s "$checkpoint/checkpoint.json" || ! -s "$checkpoint/adapter_model.safetensors" ]]; then
    echo "Missing step-0100 checkpoint in $run; no signal sent" >&2
    exit 1
fi
if [[ -e "$lock" ]] && ! flock -n "$lock" true; then
    command -v fuser >/dev/null
    fuser -k -TERM "$lock" || true
fi
if ! flock -w 60 "$lock" true; then
    echo "Training lock is still held: $lock" >&2
    exit 1
fi
echo "Training lock is free. Saved checkpoint: $checkpoint"

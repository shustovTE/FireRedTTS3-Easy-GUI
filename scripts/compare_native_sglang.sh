#!/usr/bin/env bash
# Six listening outputs: two whole texts x native / SGLang WAV / SGLang stream.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
compare_container="${1:-sglang-fire-tts-firered-tts3-1}"
compare_voice="${2:-Dovbish_Anastasiia}"
compare_release="${COMPARE_RELEASE:-training/exports/FireRedTTS3-Base-Russian-20s-07}"
compare_seed="${DIAG_SEED:-1234}"
compare_threshold="${DIAG_STOP_THRESHOLD:-0.5}"
test -f "$compare_release/release_manifest.json"
test -x .venv/bin/python
mkdir -p training/eval
compare_output=$(mktemp -d "$PWD/training/eval/native-sglang-07-XXXXXXXX")
compare_remote="/tmp/$(basename "$compare_output")"
mkdir "$compare_output/texts"
cp examples/diagnostics/long-cities.txt "$compare_output/texts/cities.txt"
cp examples/diagnostics/long-timofey.txt "$compare_output/texts/timofey.txt"
printf 'Comparison output: %s\n' "$compare_output"

docker exec -i -e COMPARE_OUTPUT="$compare_remote" -e COMPARE_VOICE="$compare_voice" \
  "$compare_container" /opt/sglang/bin/python3 - <<'PY'
import base64
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from sglang_omni.serve.speech_voices import SpeakerSampleStore

store = SpeakerSampleStore(root_dir=os.environ.get("SPEAKER_SAMPLES_DIR", "/speakers"))
reference = store.resolve_reference(os.environ["COMPARE_VOICE"])
if reference is None or not reference.voice.ref_text or not reference.voice.ref_text.strip():
    raise RuntimeError("Stored voice or its reference transcript is missing")
prefix, data = reference.ref_audio.split(",", 1)
if prefix != "data:audio/wav;base64":
    raise RuntimeError("Unexpected saved reference encoding")
audio = base64.b64decode(data, validate=True)
root = Path(os.environ["COMPARE_OUTPUT"])
root.mkdir(parents=True, exist_ok=False)
(root / "reference.wav").write_bytes(audio)
(root / "reference.json").write_text(json.dumps(dict(voice=reference.voice.name,
    reference_text=reference.voice.ref_text, reference_sha256=hashlib.sha256(audio).hexdigest()),
    ensure_ascii=False, indent=2), encoding="utf-8")
model = Path(Path("/tmp/firered-model-path").read_text().strip())
for name, destination in (("release_manifest.json", "deployed_manifest.json"),
                          ("export_provenance.json", "deployed_provenance.json"),
                          ("config.json", "deployed_config.json")):
    shutil.copyfile(model / name, root / destination)
runtime = dict(model_path=str(model), commit=subprocess.check_output(
    ["git", "-C", "/opt/firered-omni", "rev-parse", "HEAD"], text=True).strip())
(root / "deployed_runtime.json").write_text(json.dumps(runtime, indent=2), encoding="utf-8")
print("Saved exact reference and deployed release metadata:", root)
PY
docker cp "$compare_container:$compare_remote/." "$compare_output/"

OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  .venv/bin/python -u -m fireredtts3_local.native_comparison native "$compare_output" \
  --release "$compare_release" --seed "$compare_seed" --stop-threshold "$compare_threshold" \
  2>&1 | tee "$compare_output/native.log"

compare_status=0
for compare_case in cities timofey; do
  DIAG_TEXT_FILE="$compare_output/texts/$compare_case.txt" \
    DIAG_OUTPUT_DIR="$compare_output/sglang/$compare_case" \
    DIAG_REFERENCE_DIR="$compare_output" \
    DIAG_SEED="$compare_seed" DIAG_STOP_THRESHOLD="$compare_threshold" \
    bash scripts/diagnose_long_speech.sh "$compare_container" "$compare_voice" \
    2>&1 | tee "$compare_output/sglang-$compare_case.log" || compare_status=$?
done
if [ "$compare_status" -ne 0 ]; then
  printf 'Some SGLang requests failed; native output and error logs are saved in %s\n' "$compare_output" >&2
  exit "$compare_status"
fi
.venv/bin/python -m fireredtts3_local.native_comparison summarize "$compare_output"

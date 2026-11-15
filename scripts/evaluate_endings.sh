#!/usr/bin/env bash
# Compare native old/new adapters with the exact WAV/text of a stored SGLang voice.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

tail_container="${1:-sglang-fire-tts-firered-tts3-1}"
tail_voice="${2:-Dovbish_Anastasiia}"
tail_output="${3:-training/eval/endings-06}"
tail_remote="/tmp/firered-endings-reference-$$"

for tail_adapter in \
  training/outputs/russian-yo-half-05/best-validation \
  training/outputs/russian-trimmed-base-06/checkpoint-step-006471; do
  test -f "$tail_adapter/adapter_model.safetensors"
done
mkdir -p "$(dirname "$tail_output")"
mkdir "$tail_output"

docker exec -i -e TAIL_REFERENCE_DIR="$tail_remote" -e TAIL_VOICE="$tail_voice" \
  "$tail_container" /opt/sglang/bin/python3 - <<'PY'
import base64
import hashlib
import json
import os
from pathlib import Path
from sglang_omni.serve.speech_voices import SpeakerSampleStore

store = SpeakerSampleStore(root_dir=os.environ.get("SPEAKER_SAMPLES_DIR", "/speakers"))
reference = store.resolve_reference(os.environ["TAIL_VOICE"])
if reference is None:
    raise RuntimeError("Named voice was not found in the container's speaker store")
text = reference.voice.ref_text
if not text or not text.strip():
    raise RuntimeError("Stored voice has no ref_text. Supply the actual reference transcript; do not guess it.")
prefix, encoded = reference.ref_audio.split(",", 1)
if prefix != "data:audio/wav;base64":
    raise RuntimeError("Unexpected reference encoding")
wav = base64.b64decode(encoded, validate=True)
root = Path(os.environ["TAIL_REFERENCE_DIR"])
root.mkdir(parents=True, exist_ok=False)
(root / "reference.wav").write_bytes(wav)
(root / "reference.json").write_text(json.dumps({
    "voice": reference.voice.name, "reference_text": text,
    "reference_sha256": hashlib.sha256(wav).hexdigest(),
}, ensure_ascii=False, indent=2), encoding="utf-8")
print("Exported stored reference:", reference.voice.name)
PY

docker cp "$tail_container:$tail_remote/." "$tail_output/"

.venv/bin/python - "$tail_output" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
reference = json.loads((root / "reference.json").read_text(encoding="utf-8"))
assert hashlib.sha256((root / "reference.wav").read_bytes()).hexdigest() == reference["reference_sha256"]
texts = [
    "Ничего страшного. Подскажите хотя бы город, чтобы я понимала, работаем ли мы в вашем регионе?",
    "Поняла, Москва. Подскажите, какая площадь под ремонт?",
    "Поняла. А расскажите, пожалуйста, какие у вас пожелания по ремонту? Может, есть какие-то важные моменты, на которые стоит обратить внимание?",
]
rows = [dict(id=f"q{number}_seed{seed}", speaker_id=reference["voice"], text=text,
             reference_audio="reference.wav", reference_text=reference["reference_text"],
             language="Russian", seed=seed)
        for number, text in enumerate(texts, 1) for seed in (1234, 1235, 1236)]
(root / "benchmark.jsonl").write_text(
    "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
(root / "settings.json").write_text(json.dumps({
    "backend": "native sdpa, torch_compile=False", "stop_threshold": 0.5,
    "n_timesteps": 10, "inference_cfg": 2.0, "do_split": False,
    "note": "Native generation control; not a validation of the merged model in SGLang.",
}, ensure_ascii=False, indent=2), encoding="utf-8")
PY

OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  .venv/bin/python -u -m fireredtts3_local.adapt evaluate \
  "$tail_output/benchmark.jsonl" \
  --adapter training/outputs/russian-yo-half-05/best-validation \
  --output "$tail_output/old-05"

OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  .venv/bin/python -u -m fireredtts3_local.adapt evaluate \
  "$tail_output/benchmark.jsonl" \
  --adapter training/outputs/russian-trimmed-base-06/checkpoint-step-006471 \
  --output "$tail_output/new-06"

.venv/bin/python - "$tail_output" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
for label in ("old-05", "new-06"):
    folder = root / label
    report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
    assert report["complete"]
    for row in report["results"]:
        name = row["id"] + ".wav"
        assert Path(name).name == name
        (folder / row["audio"]).rename(folder / name)
        row["audio"] = name
        print(f"{label}/{name}: {row['duration']:.2f}s")
    (folder / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print("Saved:", root.resolve())
PY

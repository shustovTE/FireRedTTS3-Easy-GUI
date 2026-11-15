#!/usr/bin/env bash
# Save direct SGLang responses for a long request and its individual sentences.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

diag_container="${1:-sglang-fire-tts-firered-tts3-1}"
diag_voice="${2:-Dovbish_Anastasiia}"
diag_text_file="${DIAG_TEXT_FILE:-}"
if [ -n "$diag_text_file" ] && [ ! -f "$diag_text_file" ]; then
  printf 'Text file not found: %s\n' "$diag_text_file" >&2
  exit 2
fi
mkdir -p training/eval
diag_output=$(mktemp -d "$PWD/training/eval/long-speech-XXXXXXXX")
diag_remote="/tmp/$(basename "$diag_output")"
diag_remote_text=""
if [ -n "$diag_text_file" ]; then
  diag_remote_text="$diag_remote.input.txt"
  docker cp "$diag_text_file" "$diag_container:$diag_remote_text"
fi
diag_status=0

docker exec -i -e DIAG_OUTPUT="$diag_remote" -e DIAG_VOICE="$diag_voice" \
  -e DIAG_SEED="${DIAG_SEED:-1234}" \
  -e DIAG_STOP_THRESHOLD="${DIAG_STOP_THRESHOLD:-0.5}" \
  -e DIAG_WHOLE_TEXT="${DIAG_WHOLE_TEXT:-0}" \
  -e DIAG_READY_TIMEOUT="${DIAG_READY_TIMEOUT:-1800}" \
  -e DIAG_INPUT_FILE="$diag_remote_text" \
  "$diag_container" /opt/sglang/bin/python3 -u - <<'PY' || diag_status=$?
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
import wave
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import ProxyHandler, Request, build_opener

root = Path(os.environ["DIAG_OUTPUT"])
root.mkdir(parents=True, exist_ok=False)
api = "http://127.0.0.1:8092"
http = build_opener(ProxyHandler({}))
voice = os.environ["DIAG_VOICE"]
seed = int(os.environ["DIAG_SEED"])
threshold = float(os.environ["DIAG_STOP_THRESHOLD"])
if not 0 < threshold < 1:
    raise ValueError("DIAG_STOP_THRESHOLD must be between 0 and 1")

def save_json(name, value):
    (root / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")

sentences = [
    "Смотрите, раз площадь большая, то даже при косметическом ремонте важно всё чётко спланировать, чтобы не выбиться из бюджета и сроков.",
    "Мы как раз компания полного цикла, поэтому можем взять на себя всё: от подбора материалов до самой отделки.",
    "Чтобы вы понимали итоговую стоимость и состав работ именно для вашего объекта, предлагаю встретиться с нашим инженером-сметчиком.",
    "Он поможет определиться с деталями и составит точный расчёт.",
    "Когда вам было бы удобно встретиться?",
]
input_file = os.environ.get("DIAG_INPUT_FILE", "")
if input_file:
    # Preserve paragraph breaks, punctuation and wording in a single request.
    # No guessed sentence segmentation for arbitrary user-supplied text.
    original = Path(input_file).read_text(encoding="utf-8-sig").strip()
    if not original or len(original) > 2000:
        raise ValueError("Custom input must contain 1..2000 characters")
    sentences = []
    whole_text = "1"
    cases = [("long", original)]
else:
    original = " ".join(sentences)
    # Whole text requires lifting the runtime's 400-character guard first.
    whole_text = os.environ.get("DIAG_WHOLE_TEXT", "0")
    if whole_text not in {"0", "1"}:
        raise ValueError("DIAG_WHOLE_TEXT must be 0 or 1")
    cases = [("long", original if whole_text == "1" else " ".join(sentences[:3]))]
    cases.extend((f"sentence-{i:02d}", text) for i, text in enumerate(sentences, 1))
    assert len(original) == 470 and len(cases[0][1]) == (470 if whole_text == "1" else 371)
(root / "original_text.txt").write_text(original + "\n", encoding="utf-8")
settings = {
    "voice": voice, "seed": seed, "stop_threshold": threshold,
    "original_characters": len(original), "long_characters": len(cases[0][1]),
    "whole_text": whole_text == "1",
    "custom_text": bool(input_file),
    "note": "Sequential direct requests. Identical seeds do not guarantee identical audio across stream modes. No resampling, normalization or added silence.",
}
try:
    settings["runtime_commit"] = subprocess.check_output(
        ["git", "-C", "/opt/firered-omni", "rev-parse", "HEAD"],
        text=True, stderr=subprocess.DEVNULL, timeout=10).strip()
except (OSError, subprocess.SubprocessError) as error:
    settings["runtime_commit_error"] = str(error)
try:
    model_path = Path(Path("/tmp/firered-model-path").read_text().strip())
    settings["model_path"] = str(model_path)
    settings["release_id"] = json.loads((model_path / "release_manifest.json").read_text())["release_id"]
except (OSError, ValueError, KeyError) as error:
    settings["release_metadata_error"] = str(error)
try:
    runtime_source = Path("/opt/firered-omni/sglang_omni/models/firered_tts3/preprocessing.py")
    limit = re.search(r"(?m)^MAX_TEXT_CHARS = ([0-9]+)$", runtime_source.read_text())
    settings["runtime_source_text_limit"] = int(limit.group(1)) if limit else None
except OSError as error:
    settings["runtime_source_error"] = str(error)
save_json("settings.json", settings)

ready_timeout = int(os.environ.get("DIAG_READY_TIMEOUT", "1800"))
if ready_timeout < 1:
    raise ValueError("DIAG_READY_TIMEOUT must be positive")
deadline, next_notice = time.monotonic() + ready_timeout, 0.
while True:
    try:
        with http.open(api + "/health", timeout=5) as response:
            if response.status != 200:
                raise RuntimeError(f"Health HTTP {response.status}")
        break
    except Exception:
        if time.monotonic() >= deadline:
            raise RuntimeError("SGLang health timeout; inspect container logs") from None
        if time.monotonic() >= next_notice:
            print("Waiting for SGLang /health after startup...", flush=True)
            next_notice = time.monotonic() + 30
        time.sleep(5)

with http.open(api + "/v1/audio/voices", timeout=30) as response:
    voices = json.load(response)
if voice not in voices.get("voices", []):
    raise RuntimeError(f"Stored voice {voice!r} not found")

def read_wav(path):
    with wave.open(str(path), "rb") as wav:
        params = (wav.getnchannels(), wav.getsampwidth(), wav.getframerate())
        frames = wav.getnframes()
        data = wav.readframes(frames)
    if frames <= 0 or len(data) != frames * params[0] * params[1]:
        raise ValueError(f"Empty or truncated WAV: {path.name}")
    return params, data, frames / params[2]

results = []
for case, text in cases:
    for stream in (False, True):
        mode = "stream" if stream else "full"
        name = f"{case}_{mode}"
        payload = {
            "model": "FireRedTTS3-Base", "voice": voice, "language": "Russian",
            "input": text, "seed": seed, "stream": stream, "stream_format": "audio",
            "response_format": "pcm" if stream else "wav",
            "stage_params": {"talker": {"stop_threshold": threshold}},
        }
        save_json(name + ".request.json", payload)
        row = {"name": name, "characters": len(text)}
        started = time.monotonic()
        print(f"[{len(results) + 1}/{2 * len(cases)}] {name}", flush=True)
        try:
            request = Request(api + "/v1/audio/speech",
                              data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                              headers={"Content-Type": "application/json"})
            partial = root / (name + ".part")
            with http.open(request, timeout=600) as response:
                headers = {k.lower(): v for k, v in response.headers.items()}
                save_json(name + ".headers.json", headers)
                with partial.open("wb") as output:
                    shutil.copyfileobj(response, output)
            raw = root / (name + (".pcm" if stream else ".wav"))
            partial.rename(raw)
            wav_path = root / (name + ".wav")
            if stream:
                rate = int(headers["x-sample-rate"])
                channels = int(headers["x-channels"])
                bits = int(headers["x-bit-depth"])
                if (rate <= 0 or channels != 1 or bits != 16
                        or headers.get("content-type", "").split(";")[0] != "audio/pcm"
                        or not raw.stat().st_size or raw.stat().st_size % 2):
                    raise ValueError("Unexpected PCM format; see saved headers/PCM")
                with wave.open(str(wav_path), "wb") as wav:
                    wav.setparams((channels, bits // 8, rate, 0, "NONE", "not compressed"))
                    wav.writeframes(raw.read_bytes())
            _, _, duration = read_wav(wav_path)
            row.update(wav=wav_path.name, seconds=duration,
                       sha256=hashlib.sha256(wav_path.read_bytes()).hexdigest(),
                       finish_reason=headers.get("x-finish-reason"))
            if stream:
                try:
                    speech_id = headers["x-sglang-omni-speech-id"]
                    with http.open(api + "/v1/audio/speech/" + quote(speech_id, safe=""), timeout=30) as response:
                        outcome = json.load(response)
                    save_json(name + ".outcome.json", outcome)
                    row["finish_reason"] = outcome.get("finish_reason")
                except Exception as error:
                    row["outcome_error"] = str(error)
            print(f"  {duration:.2f}s; finish={row['finish_reason']}", flush=True)
        except Exception as error:
            detail = str(error)
            if isinstance(error, HTTPError):
                detail += "\n" + error.read(4096).decode("utf-8", errors="replace")
            row["error"] = detail
            (root / (name + ".error.txt")).write_text(detail, encoding="utf-8")
            print(f"  ERROR: {detail}", flush=True)
        row["elapsed_seconds"] = time.monotonic() - started
        results.append(row)
        save_json("summary.json", results)

if sentences and not any("error" in row for row in results):
    for mode in ("full", "stream"):
        params = None
        audio = bytearray()
        boundaries = []
        for i in range(1, 6):
            name = f"sentence-{i:02d}_{mode}.wav"
            current, data, duration = read_wav(root / name)
            if params is not None and params != current:
                raise ValueError("Sentence WAV formats differ; refusing to resample")
            params = current
            boundaries.append({"file": name, "start_seconds": len(audio) / (params[0] * params[1] * params[2]),
                               "seconds": duration})
            audio.extend(data)
            if i in (3, 5):
                joined = f"sentences-{i}_joined_{mode}"
                with wave.open(str(root / (joined + ".wav")), "wb") as wav:
                    wav.setparams((*params, 0, "NONE", "not compressed"))
                    wav.writeframes(audio)
                save_json(joined + ".boundaries.json", boundaries)
print(f"Saved: {root}", flush=True)
raise SystemExit(1 if any("error" in row for row in results) else 0)
PY

docker cp "$diag_container:$diag_remote/." "$diag_output/"
printf '\nFiles saved on the host: %s\n' "$diag_output"
exit "$diag_status"

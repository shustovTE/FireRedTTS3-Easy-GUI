#!/usr/bin/env bash
# Patch only the request text guard in the existing container; restart separately.
set -euo pipefail
text_limit="${1:-2000}"
text_container="${2:-sglang-fire-tts-firered-tts3-1}"

docker exec -i -e FIRERED_TEXT_LIMIT="$text_limit" \
  -e FIRERED_RUNTIME_DIR="${FIRERED_RUNTIME_DIR:-/opt/firered-omni}" \
  "$text_container" /opt/sglang/bin/python3 - <<'PY'
import os
from pathlib import Path
import re
import tempfile

limit = int(os.environ["FIRERED_TEXT_LIMIT"])
if not 1 <= limit <= 2000:
    raise ValueError("Diagnostic text limit must be 1..2000")
path = Path(os.environ["FIRERED_RUNTIME_DIR"]) / "sglang_omni/models/firered_tts3/preprocessing.py"
source = path.read_text(encoding="utf-8")
pattern = r"(?m)^MAX_TEXT_CHARS = ([0-9]+)$"
matches = list(re.finditer(pattern, source))
if len(matches) != 1:
    raise ValueError("Expected one literal MAX_TEXT_CHARS assignment; inspect this runtime before patching")
old = int(matches[0].group(1))
updated = re.sub(pattern, f"MAX_TEXT_CHARS = {limit}", source)
compile(updated, str(path), "exec")
if updated != source:
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(updated)
    try:
        temporary.chmod(path.stat().st_mode & 0o777)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
print(f"{path}: MAX_TEXT_CHARS {old} -> {limit}")
print("Source updated. Restart the existing container to reload worker processes.")
print("This diagnostic patch survives docker restart, but not container recreation.")
PY

printf '\nNext: docker restart %s\n' "$text_container"

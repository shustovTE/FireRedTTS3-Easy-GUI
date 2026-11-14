"""Validated, speaker-aware data for conditional speech training.

An utterance is never cropped here: audio and transcript are an indivisible pair.
Long recordings must be segmented with aligned transcripts during preparation.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import re
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import soundfile as sf


def file_digest(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def model_identity(base: str | Path) -> str:
    """Bind adapters/resume/cache to actual weights and tokenizer, not a repo name."""
    base = Path(base)
    digest = hashlib.sha256()
    for folder in ("fireredtts3_base", "redae", "campp", "text_tokenizer"):
        files = sorted(p for p in (base / folder).glob("*") if p.is_file() and p.suffix in {".json", ".safetensors", ".bin", ".txt"})
        if not files:
            raise ValueError(f"Missing base model artifacts: {folder}")
        for path in files:
            digest.update(str(path.relative_to(base)).replace("\\", "/").encode())
            digest.update(file_digest(path).encode())
    return digest.hexdigest()


def plain_text(text: str) -> str:
    # RUAccent's '+' precedes a stressed vowel. Preserve arithmetic '+' signs.
    text = re.sub(r"\+(?=[АЕЁИОУЫЭЮЯаеёиоуыэюя])", "", str(text))
    return " ".join(text.replace("\u0301", "").split())


@lru_cache(maxsize=128)
def language_name(language: str) -> str:
    from .model_manager import ModelManager
    ModelManager._ensure_upstream_namespace()
    from fireredtts3.utils.text_tokenizer import MULTI_LANG_TAGS, MULTI_DIALECT_TAGS
    language = {"ru": "Russian", "en": "English", "zh": "Chinese"}.get(language, language)
    if f"<|{language}|>" not in MULTI_LANG_TAGS + MULTI_DIALECT_TAGS:
        raise ValueError(f"Unsupported explicit training language: {language!r}")
    return language


def write_manifest(path: str | Path, rows: list[dict]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)
    return path


def load_manifest(path: str | Path) -> list[dict]:
    path = Path(path).resolve()
    rows = []
    seen_ids = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        for key in ("audio", "text", "speaker_id", "language"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError(f"{path.name}:{line_number}: missing {key}. Supply speaker metadata; do not guess identities.")
        audio = Path(row["audio"])
        audio = audio if audio.is_absolute() else path.parent / audio
        audio = audio.resolve()
        if not audio.is_file():
            raise ValueError(f"Missing audio: {audio}")
        row["audio"] = str(audio)
        row["language"] = language_name(row["language"])
        row["text"] = plain_text(row["text"])
        if not row["text"]:
            raise ValueError(f"Empty transcript: {audio}")
        info = sf.info(audio)
        if info.channels != 1 or info.frames == 0 or not math.isfinite(info.duration):
            raise ValueError(f"Training audio must be nonempty mono: {audio}")
        row["duration"] = float(info.duration)
        if "start" in row or "end" in row:
            start, end = row.get("start"), row.get("end")
            if not isinstance(start, (int, float)) or not isinstance(end, (int, float)) or not math.isfinite(start) or not math.isfinite(end) or not 0 <= start < end:
                raise ValueError(f"Invalid source time span: {audio}")
            if abs(end - start - info.duration) > 0.15:
                raise ValueError(f"Source span/audio duration mismatch: {audio}")
        row["audio_sha256"] = file_digest(audio)
        row.setdefault("utt_id", row["audio_sha256"])
        if row["utt_id"] in seen_ids:
            raise ValueError(f"Duplicate utterance id: {row['utt_id']}")
        seen_ids.add(row["utt_id"])
        rows.append(row)
    if not rows:
        raise ValueError(f"Empty manifest: {path}")
    return rows


def identity_keys(row: dict, scope: str = "speaker") -> set[str]:
    keys = {"audio:" + row["audio_sha256"]}
    if row.get("source_id"):
        keys.add("source:" + str(row["source_id"]))
    if scope == "speaker":
        keys.add("speaker:" + row["speaker_id"])
        # Global IDs are conservative leakage guards, not trusted pairing labels.
        if row.get("global_spk_id"):
            keys.add("global:" + str(row["global_spk_id"]))
    return keys


def assert_disjoint(train: list[dict], validation: list[dict], scope: str = "speaker"):
    if scope not in {"speaker", "recording"}:
        raise ValueError("validation_scope must be speaker or recording")
    left = {key for row in train for key in identity_keys(row, scope)}
    overlap = left.intersection(key for row in validation for key in identity_keys(row, scope))
    if overlap:
        raise ValueError(f"Train/validation leakage ({scope}): {sorted(overlap)[:5]}")


def split_manifest(rows: list[dict], fraction: float = 0.1, seed: int = 1234, scope: str = "speaker"):
    """Connected components prevent identities/recordings spanning both splits."""
    if not 0 < fraction < 1 or scope not in {"speaker", "recording"}:
        raise ValueError("Invalid split fraction/scope")
    parent = list(range(len(rows)))

    def root(i):
        while i != parent[i]:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    owners = {}
    for i, row in enumerate(rows):
        for key in identity_keys(row, scope):
            if key in owners:
                parent[root(i)] = root(owners[key])
            owners[key] = i
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[root(i)].append(row)
    groups = list(groups.values())
    if len(groups) < 2:
        raise ValueError("Cannot create disjoint validation: need more independent speakers/recordings.")
    random.Random(seed).shuffle(groups)
    wanted = max(1, round(len(rows) * fraction))
    validation = []
    while len(groups) > 1 and len(validation) < wanted:
        validation.extend(groups.pop())
    train = [row for group in groups for row in group]
    assert_disjoint(train, validation, scope)
    return train, validation


class SpeakerPairs:
    """Sample speakers uniformly, then targets, then a distinct valid reference.

    Pairing is also language-specific: the Base prompt contains one language tag.
    Duplicate files and overlapping excerpts of a source are never paired.
    """
    def __init__(self, rows, max_prompt_seconds=12.0, max_target_seconds=20.0):
        self.rows = rows
        groups = defaultdict(list)
        for row in rows:
            if not 0 < row["duration"] <= max_target_seconds:
                raise ValueError(f"Clip {row['utt_id']} is {row['duration']:.2f}s; align/segment it before training (limit {max_target_seconds}s).")
            groups[(row["speaker_id"], row["language"])].append(row)
        self.groups = {}
        rejected = []
        for key, clips in groups.items():
            refs = [r for r in clips if r["duration"] <= max_prompt_seconds]
            # Linear-size indexes; candidate rejection is handled in sample().
            usable = []
            for row in clips:
                if any(self.distinct(row, p) for p in refs):
                    usable.append(row)
                else:
                    rejected.append(row["utt_id"])
            if usable:
                self.groups[key] = (usable, refs)
        if rejected:
            raise ValueError(f"{len(rejected)} clips lack a distinct same-speaker reference within {max_prompt_seconds}s; first: {rejected[:3]}")
        if not self.groups:
            raise ValueError("No same-speaker training pairs")
        self.speakers = sorted(self.groups)
        self.balance_groups = defaultdict(list)
        for key, (clips, _) in self.groups.items():
            global_ids = {r["global_spk_id"] for r in clips if r.get("global_spk_id")}
            if len(global_ids) > 1:
                raise ValueError(f"Conflicting global speaker identities within {key}")
            identity = "global:" + next(iter(global_ids)) if global_ids else "local:" + key[0]
            self.balance_groups[(identity, key[1])].append(key)
        self.balance_ids = sorted(self.balance_groups)

    @staticmethod
    def distinct(a, b):
        if a["audio_sha256"] == b["audio_sha256"]:
            return False
        # A YO video's utterances each have their own local alignment clock.
        # source_id groups the video for splitting; segment_source_id identifies
        # the timeline on which start/end are measured.
        source_a = a.get("segment_source_id", a.get("source_id"))
        source_b = b.get("segment_source_id", b.get("source_id"))
        if source_a and source_a == source_b:
            if all(k in a and k in b for k in ("start", "end")):
                return a["end"] <= b["start"] or b["end"] <= a["start"]
        return True

    def sample(self, rng: random.Random):
        identity = rng.choice(self.balance_ids)
        targets, refs = self.groups[rng.choice(self.balance_groups[identity])]
        target = rng.choice(targets)
        for _ in range(16):
            prompt = rng.choice(refs)
            if self.distinct(prompt, target):
                return prompt, target
        return rng.choice([p for p in refs if self.distinct(p, target)]), target


def dataset_fingerprint(rows: list[dict]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(json.dumps(row, sort_keys=True, ensure_ascii=False).encode())
        digest.update(b"\n")
    return digest.hexdigest()

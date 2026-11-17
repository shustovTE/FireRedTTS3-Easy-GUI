"""Pinned, human-validated Common Voice Scripted 27 Russian recordings.

Preparation preserves complete source audio and supplied words. Text eligibility
is a conservative punctuation/capitalization proxy, not a grammatical proof.
No ASR, FireRed model, torch, audio cropping or clip-duration ceiling is used.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import getpass
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tarfile
import tempfile
import urllib.parse
import urllib.request

from filelock import FileLock
import numpy as np
import soundfile as sf

from .common_voice_download import NoRedirect, USER_AGENT, safe_failure
from .training_data import dataset_fingerprint, file_digest, plain_text, write_manifest
from .workspace_paths import local_path

DATASET_ID = "cmu5x45pn00dao107j4o9w2yv"
CORPUS = "cv-corpus-27.0-2026-09-11/ru"
FILENAME = "common-voice-scripted-speech-27-0-russia-72f7812a.tar.gz"
ARCHIVE_BYTES = 7093766588
ARCHIVE_SHA256 = "3f269caf4757a7837048153067ca84e29c53fd5dee9394a55e6ecc92afb6055a"
DEFAULT_ARCHIVE = f"training/raw/common-voice-scripted-27-ru/{FILENAME}"
DEFAULT_OUTPUT = "training/datasets/cv-scripted-27"
SCHEMA = "cv-scripted-27-v1"
METADATA = ("validated.tsv", "validated_sentences.tsv", "reported.tsv", "clip_durations.tsv")
PUNCTUATION_POLICY = (
    "Human-validated audio and matching human-validated sentence; exclude reported sentence IDs; "
    "uppercase/digit start, balanced quotes/brackets, final . ! or ? with no terminal ellipsis. "
    "This is a punctuation proxy, not proof of grammatical or semantic completeness. "
    "Words and punctuation are preserved; only whitespace is normalized. Review examples manually."
)


def _json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def lexical_key(text):
    """Same leakage key as the existing corpora; never used as training text."""
    return " ".join(re.findall(r"\w+", plain_text(text).casefold().replace("ё", "е")))


def category(text):
    if "?" in text:
        return "question"
    ending = re.sub(r'[\s»”“’"\)\]\}]+$', "", text)
    return "statement" if "!" not in text and ending.endswith(".") else "other"


def transcript(value):
    """Reject obvious incomplete forms without inventing words/punctuation."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("missing_transcript")
    text = " ".join(value.split())
    if not lexical_key(text):
        raise ValueError("missing_words")
    start = re.sub(r'^[\s«„“"\(\[\{—–-]+', "", text)
    if not start or not (start[0].isupper() or start[0].isdigit()):
        raise ValueError("incomplete_sentence_start")
    tail = re.search(r'([.!?…⋯\s»”“’"\'\)\]\}]+)$', text)
    marks = re.sub(r'[\s»”“’"\'\)\]\}]', "", tail[1]) if tail else ""
    if not marks or any(c not in ".!?" for c in marks) or marks.count(".") > 1:
        raise ValueError("incomplete_sentence_ending")
    # Directional Russian quotation marks can use „...“ or “...” styles.
    pairs, stack = {"(": ")", "[": "]", "{": "}", "«": "»", "„": "“"}, []
    for char in text:
        if stack and char == stack[-1]:
            stack.pop()
        elif char in pairs:
            stack.append(pairs[char])
        elif char == "“":
            stack.append("”")
        elif char in ")] }»”".replace(" ", ""):
            raise ValueError("unbalanced_quotes_or_brackets")
    if stack or text.count('"') % 2:
        raise ValueError("unbalanced_quotes_or_brackets")
    bare = re.sub(r'[\s»”“’"\'\)\]\}]+$', "", text)
    if re.search(r"\b(?:г|ул|д|им|стр|рис|руб|мин|сек|млн|млрд|т\.\s*[дп])\.$", bare, re.I):
        raise ValueError("ambiguous_abbreviation_ending")
    return text


def verify_archive(path):
    path = Path(path)
    if path.stat().st_size != ARCHIVE_BYTES or file_digest(path) != ARCHIVE_SHA256:
        raise ValueError("Archive differs from the pinned Common Voice Scripted 27 Russian release")
    return path


def _download_url():
    key = os.environ.get("MOZILLA_DATA_COLLECTIVE_API_KEY") or getpass.getpass("Mozilla Data Collective API key: ")
    if not key.strip():
        raise ValueError("Mozilla API key is required")
    request = urllib.request.Request(
        f"https://mozilladatacollective.com/api/datasets/{DATASET_ID}/download", data=b"{}", method="POST",
        headers={"Authorization": "Bearer " + key.strip(), "Content-Type": "application/json",
                 "Accept": "application/json", "User-Agent": USER_AGENT})
    try:
        with urllib.request.build_opener(NoRedirect()).open(request, timeout=60) as response:
            url = json.load(response).get("downloadUrl")
    except Exception as exc:
        raise safe_failure("Mozilla download API", exc) from None
    finally:
        del request, key
    parsed = urllib.parse.urlsplit(url) if isinstance(url, str) else None
    if parsed is None or parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("Mozilla API did not return an HTTPS storage URL")
    return url


def download(archive=DEFAULT_ARCHIVE, log=print):
    """Resume a bounded .part file; install only after pinned size/SHA256 checks.

    A fresh signed URL is requested on each invocation and never persisted. The
    bearer credential is sent only to Mozilla, with API redirects disabled.
    """
    archive = Path(archive).resolve()
    archive.parent.mkdir(parents=True, exist_ok=True)
    partial = archive.with_name(archive.name + ".part")
    with FileLock(str(archive) + ".lock", timeout=0):
        if archive.exists():
            verify_archive(archive)
            log(f"[cv-scripted] Verified existing archive: {archive}")
            return archive
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > ARCHIVE_BYTES:
            raise ValueError("Partial archive is oversized; remove the .part file before retrying")
        if offset == ARCHIVE_BYTES:
            verify_archive(partial)
            partial.replace(archive)
            return archive
        url = _download_url()
        headers = {"User-Agent": USER_AGENT, "Accept-Encoding": "identity"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        request = urllib.request.Request(url, headers=headers)
        try:
            # No API Authorization header is forwarded to the storage service.
            with urllib.request.urlopen(request, timeout=90) as response:
                status = response.status
                if status == 206:
                    expected = f"bytes {offset}-{ARCHIVE_BYTES-1}/{ARCHIVE_BYTES}"
                    if response.headers.get("Content-Range") != expected:
                        raise ValueError("Storage returned an unexpected partial range")
                elif status == 200:
                    offset = 0  # Server ignored Range: replace the partial, never append.
                else:
                    raise ValueError("Storage returned an unexpected response status")
                size = response.headers.get("Content-Length")
                if size is not None and int(size) != ARCHIVE_BYTES - offset:
                    raise ValueError("Storage size differs from the pinned release")
                received, next_log = offset, offset
                with partial.open("ab" if offset else "wb") as out:
                    while block := response.read(4 * 1024 * 1024):
                        received += len(block)
                        if received > ARCHIVE_BYTES:
                            raise ValueError("Oversized archive")
                        out.write(block)
                        if received >= next_log or received == ARCHIVE_BYTES:
                            log(f"[cv-scripted] Downloaded {received/1024**3:.2f}/{ARCHIVE_BYTES/1024**3:.2f} GiB")
                            next_log = received + 128 * 1024**2
        except Exception as exc:
            raise safe_failure("Common Voice storage download (partial retained; rerun to resume)", exc) from None
        finally:
            del request, url
        verify_archive(partial)
        partial.replace(archive)
    return archive


def _safe_member(member):
    path = PurePosixPath(member.name)
    root = PurePosixPath(CORPUS)
    if (path.is_absolute() or ".." in path.parts or "\\" in member.name or not path.parts
            or not (member.isfile() or member.isdir())
            or not (path == root or path in root.parents or root in path.parents)):
        raise ValueError("Unsafe or unexpected archive member")
    return path


def _copy_member(source, member, destination):
    if not member.isfile() or not 0 < member.size <= 2 * 1024**3:
        raise ValueError("Unexpected selected archive member size/type")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.extractfile(member) as src, destination.open("xb") as dst:
        shutil.copyfileobj(src, dst, length=1024 * 1024)
    if destination.stat().st_size != member.size:
        raise ValueError("Truncated archive member")


def extract_metadata(archive, destination):
    """Read only the four required metadata files; caller verifies the archive."""
    destination = Path(destination)
    wanted = {f"{CORPUS}/{name}": name for name in METADATA}
    seen = set()
    with tarfile.open(archive, "r|gz") as source:
        for member in source:
            _safe_member(member)
            if member.name not in wanted:
                continue
            if member.name in seen or member.size > 256 * 1024**2:
                raise ValueError("Duplicate or oversized metadata member")
            _copy_member(source, member, destination / wanted[member.name])
            seen.add(member.name)
            if seen == set(wanted):
                break
    if seen != set(wanted):
        raise ValueError("Archive is missing required validated metadata")
    return destination


def _tsv(path, required):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if not set(required) <= set(reader.fieldnames or []):
            raise ValueError(f"Unexpected metadata schema: {Path(path).name}")
        yield from reader


def candidates(metadata):
    """Only confirmed audio with the exact confirmed sentence, never bare ASR."""
    metadata = Path(metadata)
    sentences, invalid_sentence_ids = {}, set()
    for row in _tsv(metadata / "validated_sentences.tsv", {"sentence_id", "sentence"}):
        identity, text = row["sentence_id"], " ".join((row["sentence"] or "").split())
        if not identity or identity in sentences and sentences[identity] != text:
            invalid_sentence_ids.add(identity)
        sentences[identity] = text
    reported = {r["sentence_id"] for r in _tsv(metadata / "reported.tsv", {"sentence_id"})}
    durations = {}
    for row in _tsv(metadata / "clip_durations.tsv", {"clip", "duration[ms]"}):
        try:
            duration = float(row["duration[ms]"]) / 1000
            if not math.isfinite(duration) or duration <= 0 or row["clip"] in durations:
                duration = None
        except (ValueError, TypeError):
            duration = None
        durations[row["clip"]] = duration
    rows, exclusions, paths = [], [], set()
    required = {"client_id", "path", "sentence_id", "sentence", "up_votes", "down_votes", "locale"}
    for number, row in enumerate(_tsv(metadata / "validated.tsv", required), 2):
        name = row.get("path") or ""
        try:
            if None in row or any(row.get(key) is None for key in required):
                raise ValueError("malformed_metadata_row")
            if not re.fullmatch(r"common_voice_ru_[0-9]+\.mp3", name):
                raise ValueError("invalid_audio_path")
            if name in paths:
                raise ValueError("duplicate_audio_path")
            paths.add(name)
            speaker = row["client_id"]
            if not speaker:
                raise ValueError("missing_speaker")
            if not re.fullmatch(r"[A-Za-z0-9_-]{1,256}", speaker):
                raise ValueError("invalid_speaker_identity")
            if row["locale"] != "ru":
                raise ValueError("non_russian_locale")
            sentence_id = row["sentence_id"]
            if sentence_id in reported:
                raise ValueError("reported_sentence")
            text = " ".join(row["sentence"].split())
            if sentence_id not in sentences or sentence_id in invalid_sentence_ids:
                raise ValueError("sentence_not_human_validated")
            if sentences[sentence_id] != text:
                raise ValueError("validated_sentence_mismatch")
            text = transcript(text)
            votes = [int(row[key]) for key in ("up_votes", "down_votes")]
            if min(votes) < 0:
                raise ValueError("invalid_vote_metadata")
            if not durations.get(name):
                raise ValueError("missing_or_invalid_duration")
            identity = "cv-scripted27:" + Path(name).stem
            rows.append(dict(text=text, language="Russian", speaker_id="cv-scripted27:" + speaker,
                             utt_id=identity, source_id=identity, duration=durations[name],
                             category=category(text), cv_source_path=name, cv_sentence_id=sentence_id,
                             cv_validated=True, cv_up_votes=votes[0], cv_down_votes=votes[1],
                             cv_original_transcription=row["sentence"], cv_original_split="validated"))
        except (ValueError, TypeError) as exc:
            reason = str(exc) if isinstance(exc, ValueError) and re.fullmatch(r"[a-z_]+", str(exc)) else "invalid_metadata_value"
            exclusions.append(dict(line=number, source_path=name, reason=reason))
    metadata_hashes = {name: file_digest(metadata / name) for name in METADATA}
    return rows, exclusions, metadata_hashes


def _pair(pool, budget, order):
    ordered = sorted(pool, key=order)
    shortest_after = [float("inf")] * len(ordered)
    minimum = float("inf")
    for i in range(len(ordered)-1, -1, -1):
        shortest_after[i] = minimum
        minimum = min(minimum, ordered[i]["duration"])
    for i, first in enumerate(ordered):
        if first["duration"] + shortest_after[i] > budget + 1e-8:
            continue
        for second in ordered[i+1:]:
            if first["duration"] + second["duration"] <= budget + 1e-8:
                return [first, second]
    return []


def _rank(seed, name):
    return hashlib.sha256(f"{seed}:{name}".encode()).hexdigest()


def select(rows, seed=1234, train_minutes=30., validation_seconds=200.,
           blocked_train_texts=(), blocked_validation_texts=()):
    """Global lexical holdout, per-speaker budgets, >=2 distinct pairs per split.

    'blocked_train_texts' are forbidden in this new training split because they
    already occur in validation; 'blocked_validation_texts' analogously anchor
    existing training. Validation is additional to the training budget. At least
    two whole validation recordings are retained even when their combined length
    exceeds its soft budget; that exception is always reported.
    """
    budget = train_minutes * 60
    if not math.isfinite(budget) or budget <= 0 or not math.isfinite(validation_seconds) or validation_seconds <= 0:
        raise ValueError("Speaker budgets must be finite positive numbers")
    force_val, force_train = set(blocked_train_texts), set(blocked_validation_texts)
    if force_val & force_train:
        raise ValueError("Existing corpora already leak text between train and validation")
    grouped, excluded = defaultdict(dict), []
    for row in sorted(rows, key=lambda r: _rank(seed, r["utt_id"])):
        if not math.isfinite(row["duration"]) or row["duration"] <= 0:
            raise ValueError("Candidate duration must be finite and positive")
        key, speaker = lexical_key(row["text"]), row["speaker_id"]
        if key in grouped[speaker]:
            excluded.append(dict(utt_id=row["utt_id"], speaker_id=speaker, reason="duplicate_speaker_text"))
        else:
            grouped[speaker][key] = row
    frequencies = Counter(key for pool in grouped.values() for key in pool)
    val_keys, protected_train, anchors = set(force_val), set(force_train), {}
    # Sparse speakers first; reserve two training words before assigning shared
    # prompts globally to validation. This avoids an arbitrary hash split losing
    # almost all speakers with only four to ten recordings.
    for speaker, mapping in sorted(grouped.items(), key=lambda x: (len(x[1]), x[0])):
        pool = list(mapping.values())
        available_train = [r for r in pool if lexical_key(r["text"]) not in val_keys]
        pair = _pair(available_train, budget,
                     lambda r: (lexical_key(r["text"]) not in protected_train,
                                -frequencies[lexical_key(r["text"])], _rank(seed, r["utt_id"])))
        if not pair:
            excluded.append(dict(speaker_id=speaker, clips=len(pool), reason="insufficient_train_pair_within_speaker_budget"))
            continue
        reserved = {lexical_key(r["text"]) for r in pair}
        validation = [r for r in pool if lexical_key(r["text"]) in val_keys]
        unavailable = val_keys | protected_train | reserved
        options = sorted((r for r in pool if lexical_key(r["text"]) not in unavailable),
                         key=lambda r: (frequencies[lexical_key(r["text"])], _rank(seed, r["utt_id"])))
        if len(validation) + len(options) < 2:
            excluded.append(dict(speaker_id=speaker, clips=len(pool), reason="insufficient_independent_validation_pair"))
            continue
        desired = min(validation_seconds, sum(r["duration"] for r in pool) * .1)
        seconds = sum(r["duration"] for r in validation)
        for row in options:
            if len(validation) >= 2 and seconds >= desired:
                break
            if len(validation) >= 2 and seconds + row["duration"] > validation_seconds:
                continue
            validation.append(row)
            seconds += row["duration"]
            val_keys.add(lexical_key(row["text"]))
        protected_train.update(reserved)
        anchors[speaker] = pair
    train, validation, exceptions = [], [], []
    for speaker, pair in sorted(anchors.items()):
        pool = list(grouped[speaker].values())
        train_pool = [r for r in pool if lexical_key(r["text"]) not in val_keys]
        val_pool = [r for r in pool if lexical_key(r["text"]) in val_keys]
        selected_train = list(pair)
        used = {r["utt_id"] for r in pair}
        seconds = sum(r["duration"] for r in pair)
        for row in sorted(train_pool, key=lambda r: _rank(seed, r["utt_id"])):
            if row["utt_id"] not in used and seconds + row["duration"] <= budget + 1e-8:
                selected_train.append(row)
                seconds += row["duration"]
        val_pair = _pair(val_pool, validation_seconds, lambda r: _rank(seed, r["utt_id"]))
        if not val_pair:
            val_pair = sorted(val_pool, key=lambda r: (r["duration"], r["utt_id"]))[:2]
        selected_val = list(val_pair)
        used = {r["utt_id"] for r in val_pair}
        seconds = sum(r["duration"] for r in val_pair)
        desired = min(validation_seconds, sum(r["duration"] for r in pool) * .1)
        for row in sorted(val_pool, key=lambda r: _rank(seed, r["utt_id"])):
            if seconds >= desired:
                break
            if row["utt_id"] not in used and seconds + row["duration"] <= validation_seconds + 1e-8:
                selected_val.append(row)
                seconds += row["duration"]
        if seconds > validation_seconds + 1e-8:
            exceptions.append(dict(speaker_id=speaker, seconds=seconds, limit=validation_seconds,
                                   reason="two_whole_independent_validation_recordings"))
        train.extend(selected_train)
        validation.extend(selected_val)
    if not train or not validation:
        raise ValueError("No speakers can provide independent train and validation pairs")
    chosen = {r["utt_id"] for r in train + validation}
    # Every unselected candidate is accounted for, including budget leftovers.
    excluded_ids = {r.get("utt_id") for r in excluded}
    excluded.extend(dict(utt_id=r["utt_id"], speaker_id=r["speaker_id"], reason="split_or_speaker_budget")
                    for r in rows if r["utt_id"] not in chosen and r["utt_id"] not in excluded_ids)
    validate_selection(train, validation, train_minutes, validation_seconds, require_audio=False)
    return train, validation, excluded, exceptions


def validate_selection(train, validation, train_minutes=30., validation_seconds=200., require_audio=True):
    if not train or not validation:
        raise ValueError("Both splits must be nonempty")
    if {lexical_key(r["text"]) for r in train} & {lexical_key(r["text"]) for r in validation}:
        raise ValueError("Scripted train/validation text leakage")
    if {r["speaker_id"] for r in train} != {r["speaker_id"] for r in validation}:
        raise ValueError("Each retained speaker must have extra validation")
    seen_ids, seen_audio = set(), set()
    for name, rows in (("train", train), ("validation", validation)):
        groups = defaultdict(list)
        for row in rows:
            if row["utt_id"] in seen_ids:
                raise ValueError("Repeated source recording")
            seen_ids.add(row["utt_id"])
            groups[row["speaker_id"]].append(row)
            if require_audio:
                if row["audio_sha256"] in seen_audio:
                    raise ValueError("Repeated audio content")
                seen_audio.add(row["audio_sha256"])
        for clips in groups.values():
            if len({lexical_key(r["text"]) for r in clips}) < 2:
                raise ValueError("Each split requires independent same-speaker references")
            duration = sum(r["duration"] for r in clips)
            if name == "train" and duration > train_minutes * 60 + 1e-6:
                raise ValueError("Training speaker budget exceeded")
            if name == "validation" and duration > validation_seconds + 1e-6 and len(clips) != 2:
                raise ValueError("Validation soft-budget exception permits exactly two recordings")


def summary(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["speaker_id"]].append(row)
    return dict(clips=len(rows), hours=sum(r["duration"] for r in rows)/3600,
                speakers=len(groups), longest_seconds=max((r["duration"] for r in rows), default=0.),
                categories=dict(Counter(r["category"] for r in rows)),
                fingerprint=dataset_fingerprint(rows),
                actors={s: dict(clips=len(pool), seconds=sum(r["duration"] for r in pool)) for s, pool in sorted(groups.items())})


def make_plan(metadata, *, seed=1234, train_minutes=30., validation_seconds=200.,
              blocked_train_texts=(), blocked_validation_texts=()):
    rows, rejected, hashes = candidates(metadata)
    train, validation, excluded, exceptions = select(rows, seed, train_minutes, validation_seconds,
                                                    blocked_train_texts, blocked_validation_texts)
    report = dict(schema=SCHEMA, complete=False, metadata_only=True, dataset_id=DATASET_ID,
                  dataset_url=f"https://mozilladatacollective.com/datasets/{DATASET_ID}", license="CC0-1.0",
                  archive_sha256=ARCHIVE_SHA256, archive_bytes=ARCHIVE_BYTES, metadata_sha256=hashes,
                  text_policy=PUNCTUATION_POLICY, max_clip_seconds=None,
                  source_validated_clips=len(rows)+len(rejected), asr_used=False,
                  train_minutes_per_speaker=train_minutes, validation_seconds_per_speaker=validation_seconds,
                  validation_budget_exceptions=exceptions, eligible=summary(rows),
                  datasets={"train": summary(train), "validation": summary(validation)},
                  rejections=dict(Counter(r["reason"] for r in rejected + excluded)),
                  rejection_units={"insufficient_train_pair_within_speaker_budget": "speakers",
                                   "insufficient_independent_validation_pair": "speakers",
                                   "all_other_reasons": "recordings"}, seed=seed)
    return dict(report=report, train=train, validation=validation, exclusions=rejected + excluded)


def plan(metadata, output, **kwargs):
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("Plan output already exists")
    result = make_plan(metadata, **kwargs)
    output.mkdir(parents=True)
    _json(output / "report.json", result["report"])
    _json(output / "plan.json", {k: v for k, v in result.items() if k != "exclusions"})
    write_manifest(output / "exclusions.jsonl", result["exclusions"])
    return result["report"]


def _extract_audio(archive, destination, rows, log):
    wanted = {f"{CORPUS}/clips/{r['cv_source_path']}": r["cv_source_path"] for r in rows}
    found, remaining = set(), set(wanted)
    with tarfile.open(archive, "r|gz") as source:
        for member in source:
            _safe_member(member)
            if member.name not in wanted:
                continue
            if member.name in found:
                raise ValueError("Duplicate selected audio archive member")
            _copy_member(source, member, destination / wanted[member.name])
            found.add(member.name)
            remaining.remove(member.name)
            if len(found) % 1000 == 0:
                log(f"[cv-scripted] Extracted {len(found)}/{len(wanted)} selected recordings")
            if not remaining:
                break
    return {wanted[name] for name in remaining}


def _audio_info(path):
    info = sf.info(path)
    if info.channels != 1:
        raise ValueError("unsupported_audio_channels")
    if not info.frames or info.samplerate <= 0 or not math.isfinite(info.duration):
        raise ValueError("invalid_audio")
    peak = 0.
    for block in sf.blocks(path, blocksize=65536, dtype="float32", always_2d=True):
        if not np.isfinite(block).all():
            raise ValueError("nonfinite_audio")
        peak = max(peak, float(np.abs(block).max()))
    if peak == 0:
        raise ValueError("silent_audio")
    return info, peak


def prepare(archive=DEFAULT_ARCHIVE, output=DEFAULT_OUTPUT, *, seed=1234,
            train_minutes=30., validation_seconds=200., blocked_train_texts=(),
            blocked_validation_texts=(), log=print):
    archive, output = Path(archive).resolve(), Path(output).resolve()
    verify_archive(archive)
    if archive == output or archive.is_relative_to(output) or output.is_relative_to(archive):
        raise ValueError("Preparation output must be separate from the source archive")
    output.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output) + ".lock", timeout=0):
        if output.exists():
            raise ValueError("Preparation output exists; use audit or a new directory")
        with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
            staging = Path(temporary) / "prepared"
            staging.mkdir()
            metadata = extract_metadata(archive, staging / "metadata")
            result = make_plan(metadata, seed=seed, train_minutes=train_minutes, validation_seconds=validation_seconds,
                               blocked_train_texts=blocked_train_texts, blocked_validation_texts=blocked_validation_texts)
            requested = result["train"] + result["validation"]
            missing = _extract_audio(archive, staging / "audio", requested, log)
            splits, seen_audio, audio_excluded, inspected = {}, set(), [], 0
            for name in ("train", "validation"):
                clean = []
                for row in result[name]:
                    inspected += 1
                    if inspected % 1000 == 0:
                        log(f"[cv-scripted] Auditing audio {inspected}/{len(requested)}")
                    path = staging / "audio" / row["cv_source_path"]
                    try:
                        if row["cv_source_path"] in missing:
                            raise ValueError("missing_audio")
                        info, peak = _audio_info(path)
                        digest = file_digest(path)
                        if digest in seen_audio:
                            raise ValueError("duplicate_audio_content")
                        seen_audio.add(digest)
                        clean.append(dict(row, audio=str(output / "audio" / path.name),
                                          duration=info.duration, start=0., end=info.duration,
                                          audio_sha256=digest, cv_metadata_duration=row["duration"],
                                          cv_audio_processing=dict(method="unchanged_source_audio", sample_rate=info.samplerate,
                                                                   channels=info.channels, decoded_peak=peak)))
                    except (ValueError, RuntimeError) as exc:
                        reason = str(exc) if isinstance(exc, ValueError) else "audio_decode_failed"
                        audio_excluded.append(dict(utt_id=row["utt_id"], speaker_id=row["speaker_id"], reason=reason))
                splits[name] = clean
            # Actual decoded durations, not rounded metadata, enforce budgets.
            retained = {}
            for name in ("train", "validation"):
                groups = defaultdict(list)
                for row in splits[name]:
                    groups[row["speaker_id"]].append(row)
                retained[name] = {}
                for speaker, pool in groups.items():
                    budget = train_minutes * 60 if name == "train" else validation_seconds
                    if sum(r["duration"] for r in pool) <= budget + 1e-6:
                        chosen = pool
                    else:
                        chosen = _pair(pool, budget, lambda r: _rank(seed, r["utt_id"]))
                        if not chosen and name == "validation":
                            chosen = sorted(pool, key=lambda r: (r["duration"], r["utt_id"]))[:2]
                        chosen_ids = {r["utt_id"] for r in chosen}
                        total = sum(r["duration"] for r in chosen)
                        for row in pool:
                            if row["utt_id"] not in chosen_ids and total + row["duration"] <= budget + 1e-6:
                                chosen.append(row)
                                chosen_ids.add(row["utt_id"])
                                total += row["duration"]
                        audio_excluded.extend(dict(utt_id=r["utt_id"], speaker_id=speaker, reason="actual_duration_speaker_budget")
                                              for r in pool if r["utt_id"] not in chosen_ids)
                    if len(chosen) >= 2:
                        retained[name][speaker] = chosen
            speakers = set(retained["train"]) & set(retained["validation"])
            final = {name: [r for s in sorted(speakers) for r in retained[name][s]] for name in splits}
            selected = {r["utt_id"] for rows in final.values() for r in rows}
            already = {r.get("utt_id") for r in audio_excluded}
            audio_excluded.extend(dict(utt_id=r["utt_id"], speaker_id=r["speaker_id"], reason="insufficient_pair_after_audio_audit")
                                  for rows in splits.values() for r in rows if r["utt_id"] not in selected and r["utt_id"] not in already)
            validate_selection(final["train"], final["validation"], train_minutes, validation_seconds)
            keep_paths = {r["cv_source_path"] for rows in final.values() for r in rows}
            for path in (staging / "audio").glob("*"):
                if path.name not in keep_paths:
                    path.unlink()
            for name, rows in final.items():
                write_manifest(staging / f"{name}.jsonl", rows)
            report = dict(result["report"], complete=True, metadata_only=False,
                          planned_datasets=result["report"]["datasets"], datasets={name: summary(rows) for name, rows in final.items()},
                          audio_processing="Original complete recordings, unchanged samples; no ASR, crop, resampling or VAD here.")
            report["validation_budget_exceptions"] = [dict(speaker_id=s, seconds=sum(r["duration"] for r in retained["validation"][s]),
                limit=validation_seconds, reason="two_whole_independent_validation_recordings") for s in sorted(speakers)
                if sum(r["duration"] for r in retained["validation"][s]) > validation_seconds + 1e-6]
            excluded = result["exclusions"] + audio_excluded
            report["rejections"] = dict(Counter(r["reason"] for r in excluded))
            write_manifest(staging / "exclusions.jsonl", excluded)
            _json(staging / "report.json", report)
            staging.replace(output)
    return report


def audit(output=DEFAULT_OUTPUT):
    output = local_path(output).resolve()
    report = json.loads((output / "report.json").read_text())
    if report.get("schema") != SCHEMA or not report.get("complete") or report.get("archive_sha256") != ARCHIVE_SHA256:
        raise ValueError("Expected complete pinned Scripted 27 preparation")
    if (output / "audio").is_symlink():
        raise ValueError("Prepared audio directory must not be a symlink")
    if set(report.get("metadata_sha256", {})) != set(METADATA):
        raise ValueError("Prepared metadata inventory changed")
    for name, digest in report["metadata_sha256"].items():
        if name not in METADATA or file_digest(output / "metadata" / name) != digest:
            raise ValueError("Prepared metadata changed")
    splits = {}
    for name in ("train", "validation"):
        rows = [json.loads(line) for line in (output / f"{name}.jsonl").read_text().splitlines() if line.strip()]
        if dataset_fingerprint(rows) != report["datasets"][name]["fingerprint"]:
            raise ValueError("Prepared manifest changed")
        for row in rows:
            path = local_path(row["audio"])
            if not path.resolve().is_relative_to(output / "audio") or not path.is_file() or path.is_symlink():
                raise ValueError("Prepared audio path changed")
            if file_digest(path) != row["audio_sha256"] or abs(sf.info(path).duration - row["duration"]) > 1e-6:
                raise ValueError("Prepared audio changed")
        splits[name] = rows
    validate_selection(splits["train"], splits["validation"], report["train_minutes_per_speaker"], report["validation_seconds_per_speaker"])
    return report


def _anchor_keys(paths):
    result = set()
    for path in paths:
        for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
            if line.strip():
                result.add(lexical_key(json.loads(line)["text"]))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    fetch = sub.add_parser("download")
    fetch.add_argument("--archive", default=DEFAULT_ARCHIVE)
    inspect = sub.add_parser("plan", help="Plan from extracted metadata, without audio/model/GPU loading")
    inspect.add_argument("--metadata", required=True)
    inspect.add_argument("--output", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--archive", default=DEFAULT_ARCHIVE)
    prep.add_argument("--output", default=DEFAULT_OUTPUT)
    check = sub.add_parser("audit")
    check.add_argument("--output", default=DEFAULT_OUTPUT)
    for command in (inspect, prep):
        command.add_argument("--seed", type=int, default=1234)
        command.add_argument("--train-minutes", type=float, default=30.)
        command.add_argument("--validation-seconds", type=float, default=200.)
        command.add_argument("--anchor-train", nargs="*", default=[], help="Existing training manifests whose texts cannot enter validation")
        command.add_argument("--anchor-validation", nargs="*", default=[], help="Existing validation manifests whose texts cannot enter training")
    args = vars(parser.parse_args())
    command = args.pop("command")
    if command in {"plan", "prepare"}:
        args["blocked_validation_texts"] = _anchor_keys(args.pop("anchor_train"))
        args["blocked_train_texts"] = _anchor_keys(args.pop("anchor_validation"))
    result = {"download": download, "prepare": prepare, "audit": audit, "plan": plan}[command](**args)
    print(json.dumps(str(result) if isinstance(result, Path) else result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

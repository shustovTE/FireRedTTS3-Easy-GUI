"""Confirmed Russian spontaneous speech, whole <=30s recordings, after Dialogs-08."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
import random
import re
import shutil
import tempfile

from filelock import FileLock
import numpy as np
import soundfile as sf
import torch
import torchaudio

from .common_voice_download import (ARCHIVE_SHA256, CORPUS, DATASET_ID,
                                   DEFAULT_ARCHIVE, download, extract_verified)
from .dialogs_dataset import category, lexical_key
from .training_data import (SpeakerPairs, assert_disjoint, dataset_fingerprint,
                            file_digest, plain_text, write_manifest)
from .yo_dataset import has_sentence_ending

SCHEMA = "cv-spontaneous-dialogs-v1"
DEFAULT_OUTPUT = "training/datasets/cv-spontaneous-09"
DEFAULT_RECIPE = "recipes/russian_spontaneous_96gb.json"
MANIFESTS = {"train": "train.jsonl", "validation": "validation.jsonl",
             "dialogs_train": "dialogs-train.jsonl", "dialogs_validation": "dialogs-validation.jsonl"}
ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors", "firered_lora_metadata.json")


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def transcript(text):
    """Use only supplied words. Pause labels become punctuation, not speech tokens."""
    changes = []
    def replace(match):
        tag = match[1].strip()
        if tag.casefold() in {"pause", "dramatic pause", "silent", "silence"}:
            left, right = text[:match.start()].rstrip(), text[match.end():].lstrip()
            if (match.start() and match.end() < len(text)
                    and text[match.start()-1].isalnum() and text[match.end()].isalnum()):
                raise ValueError("pause_inside_word")
            replacement = " … " if left and right and re.search(r"\w", right) else " "
        elif tag.casefold().startswith("disfluency|"):
            replacement = tag.split("|", 1)[1].strip()
            if not re.fullmatch(r"[А-Яа-яЁё][А-Яа-яЁё —-]{0,39}", replacement):
                raise ValueError("ambiguous_disfluency")
        else:
            # No inferred words for unclear/noise/laugh/unqualified disfluency.
            raise ValueError("unsupported_or_unclear_annotation")
        changes.append({"source": match[0], "replacement": replacement.strip()})
        return replacement
    result = re.sub(r"\[([^\[\]]*)\]", replace, text)
    if any(char in result for char in "[]<>"):
        raise ValueError("unsupported_annotation")
    result = plain_text(result)
    if not lexical_key(result) or not has_sentence_ending(result):
        raise ValueError("incomplete_sentence_ending")
    if re.search(r"[A-Za-z]", result):
        raise ValueError("non_russian_script")
    return result, changes


def collect(source, audio_output, blocked_texts=(), *, original_for_vad=False):
    """Audit every confirmed candidate before selection; never crop or use ASR."""
    source, audio_output = Path(source), Path(audio_output)
    audio_output.mkdir(parents=True)
    with (source / "ss-corpus-ru.tsv").open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        required = {"audio_id", "audio_file", "client_id", "duration_ms", "language",
                    "transcription", "votes", "split", "quality_tags"}
        if not required <= set(reader.fieldnames or []):
            raise ValueError("Unexpected Common Voice metadata schema")
        records = list(reader)
    reported = set()
    with (source / "ss-reported-audios-ru.tsv").open(encoding="utf-8-sig", newline="") as stream:
        reported.update(r["audio_id"] for r in csv.DictReader(stream, delimiter="\t"))
    rows, excluded, seen_ids, seen_audio, seen_text = [], [], set(), set(), set()
    blocked_texts = set(blocked_texts)
    for record in records:
        audio_id = record["audio_id"]
        if audio_id in seen_ids:
            raise ValueError("Duplicate audio ID in source metadata")
        seen_ids.add(audio_id)
        try:
            if (not record["transcription"].strip() or int(record["votes"]) < 1
                    or record["split"] not in {"dev", "test"}):
                raise ValueError("not_human_confirmed")
            if audio_id in reported:
                raise ValueError("reported_audio")
            if record["language"] != "Russian":
                raise ValueError("language")
            if not re.fullmatch(r"[0-9]+", audio_id) or not re.fullmatch(r"[a-f0-9]{32}", record["client_id"]):
                raise ValueError("invalid_source_identity")
            if record["audio_file"] != f"spontaneous-speech-ru-{audio_id}.mp3":
                raise ValueError("invalid_audio_path")
            duration = float(record["duration_ms"]) / 1000
            if not math.isfinite(duration) or duration <= 0 or (not original_for_vad and not 2 <= duration <= 30):
                raise ValueError("duration")
            if set(record["quality_tags"].split("|")) & {"transcription-length", "speech-rate", "short-audio", "long-audio"}:
                raise ValueError("source_quality_flag")
            text, changes = transcript(record["transcription"])
            key = lexical_key(text)
            if key in blocked_texts:
                raise ValueError("dialogs_text_overlap")
            if not original_for_vad and key in seen_text:
                raise ValueError("duplicate_word_sequence")
            path = source / "audios" / record["audio_file"]
            data, rate = sf.read(path, dtype="float32", always_2d=True)
            if data.shape[1] != 1 or not len(data) or not np.isfinite(data).all():
                raise ValueError("invalid_audio")
            actual = len(data) / rate
            if abs(actual - duration) > .15 or (not original_for_vad and not 2 <= actual <= 30):
                raise ValueError("audio_duration")
            if float(np.sqrt(np.mean(data.astype(np.float64)**2))) < 1e-5:
                raise ValueError("near_silent_audio")
            peak = float(np.abs(data).max())
            near_fullscale = float(np.mean(np.abs(data) >= .999))
            if near_fullscale > .001:
                raise ValueError("excess_near_fullscale_samples")
            processing = dict(source_sample_rate=rate, decoded_peak=peak, fraction_near_fullscale=near_fullscale)
            if original_for_vad:
                destination = audio_output / path.name
                shutil.copyfile(path, destination)
                seconds = actual
                processing["method"] = "original_mp3"
            else:
                wave = torchaudio.functional.resample(torch.from_numpy(data.T.copy()), rate, 24000)[0].numpy()
                # MP3/resampling can overshoot. Fixed attenuation preserves dynamics
                # and every sample; no hard clipping, compression or loudness target.
                resampled_peak = float(np.abs(wave).max())
                gain = .98 / resampled_peak if resampled_peak > 1. else 1.
                destination = audio_output / f"{audio_id}.wav"
                sf.write(destination, wave * gain, 24000, subtype="PCM_16")
                processing.update(resampled_peak=resampled_peak, fixed_gain=gain)
                seconds = sf.info(destination).duration
            digest = file_digest(destination)
            if digest in seen_audio:
                destination.unlink()
                raise ValueError("duplicate_audio")
            if not original_for_vad and seconds > 30:
                destination.unlink()
                raise ValueError("resampled_duration")
            seen_audio.add(digest)
            seen_text.add(key)
            rows.append(dict(audio=str(destination.resolve()), text=text, language="Russian",
                speaker_id="cv-sps5:" + record["client_id"], utt_id="cv-sps5:" + audio_id,
                source_id="cv-sps5:" + audio_id, duration=seconds, start=0., end=seconds,
                audio_sha256=digest, category=category(text), cv_audio_id=audio_id,
                cv_official_split=record["split"], cv_votes=int(record["votes"]),
                cv_original_transcription=record["transcription"], cv_transcript_changes=changes,
                cv_quality_tags=record["quality_tags"], cv_source_sha256=file_digest(path),
                cv_audio_processing=processing))
        except (ValueError, RuntimeError) as exc:
            excluded.append(dict(audio_id=audio_id, reason=str(exc)))
    return rows, excluded


def select(rows, seed=1234):
    """Exact count thirds per actor in BOTH splits; validation has its own cap."""
    groups = defaultdict(list)
    for row in rows:
        groups[row["speaker_id"]].append(row)
    train, validation, rejected = [], [], []
    for speaker, pool in sorted(groups.items()):
        pool = sorted(pool, key=lambda r: r["utt_id"])
        rng = random.Random(f"cv-thirds-v1:{seed}:{speaker}")
        questions = [r for r in pool if r["category"] == "question"]
        statements = [r for r in pool if r["category"] == "statement"]
        count = min(len(questions), len(statements), len(pool)//3)
        if count < 2:
            rejected.append(dict(speaker_id=speaker, clips=len(pool), reason="insufficient_thirds_for_train_and_validation"))
            continue
        q, s = rng.sample(questions, count), rng.sample(statements, count)
        used = {r["utt_id"] for r in q+s}
        rest = rng.sample([r for r in pool if r["utt_id"] not in used], count)
        triples = [[{**r, "selection_slot": slot, "selection_triple": i}
                    for slot, r in zip(("question", "statement", "random"), triple)]
                   for i, triple in enumerate(zip(q, s, rest))]
        rng.shuffle(triples)
        desired = max(1, round(count * .1))
        heldout, remaining, seconds = [], [], 0.
        for triple in triples:
            duration = sum(r["duration"] for r in triple)
            if len(heldout) < desired and seconds + duration <= 200.:
                heldout.append(triple)
                seconds += duration
            else:
                remaining.append(triple)
        selected, seconds = [], 0.
        for triple in remaining:
            duration = sum(r["duration"] for r in triple)
            if seconds + duration <= 1800.:
                selected.extend(triple)
                seconds += duration
        if not heldout or not selected:
            raise ValueError("Could not reserve whole train/validation triples")
        train.extend(selected)
        validation.extend(r for triple in heldout for r in triple)
    if not train or not validation:
        raise ValueError("No actors can supply independent balanced train/validation pairs")
    validate_selection(train, validation)
    return train, validation, rejected


def validate_selection(train, validation):
    assert_disjoint(train, validation, "recording")
    if {lexical_key(r["text"]) for r in train} & {lexical_key(r["text"]) for r in validation}:
        raise ValueError("Common Voice text leakage")
    if {r["speaker_id"] for r in train} != {r["speaker_id"] for r in validation}:
        raise ValueError("Each retained actor must have extra validation")
    for rows, budget in ((train, 1800.), (validation, 200.)):
        SpeakerPairs(rows, 30., 30.)
        for speaker in {r["speaker_id"] for r in rows}:
            clips = [r for r in rows if r["speaker_id"] == speaker]
            slots = Counter(r["selection_slot"] for r in clips)
            if (set(slots) != {"question", "statement", "random"} or len(set(slots.values())) != 1
                    or sum(r["duration"] for r in clips) > budget + 1e-6):
                raise ValueError("Per-actor quotas or duration budget changed")


def summary(rows):
    return dict(clips=len(rows), minutes=sum(r["duration"] for r in rows)/60,
                speakers=len({r["speaker_id"] for r in rows}), fingerprint=dataset_fingerprint(rows),
                slots=dict(Counter(r.get("selection_slot", "unspecified") for r in rows)),
                actors={speaker: dict(clips=len(clips), minutes=sum(r["duration"] for r in clips)/60)
                        for speaker in sorted({r["speaker_id"] for r in rows})
                        for clips in [[r for r in rows if r["speaker_id"] == speaker]]})


def audit(output=DEFAULT_OUTPUT):
    from .training_run import initial_adapter_info, prepare_run_data
    output = Path(output).resolve()
    report = json.loads((output / "report.json").read_text())
    if report.get("schema") != SCHEMA or not report.get("complete"):
        raise ValueError("Complete spontaneous preparation first")
    if file_digest(output / "recipe.json") != report["recipe_sha256"]:
        raise ValueError("Prepared recipe changed")
    config = json.loads((output / "recipe.json").read_text())
    datasets = prepare_run_data(config["manifest_path"], config["options"], config["seed"], include_adaptation_validation=True)
    for name, pairs in zip(MANIFESTS, datasets):
        if pairs is None or dataset_fingerprint(pairs.rows) != report["datasets"][name]["fingerprint"]:
            raise ValueError(f"Prepared {name} audio/metadata changed")
    validate_selection(datasets[0].rows, datasets[1].rows)
    identity = initial_adapter_info(config["initial_adapter"], config["rank"], config["alpha"], config["dropout"], config["options"]["profile"])
    if identity["fingerprint"] != report["initial_adapter_fingerprint"]:
        raise ValueError("Prepared Dialogs-08 adapter changed")
    return report


def prepare(archive=DEFAULT_ARCHIVE, output=DEFAULT_OUTPUT,
            dialogs="training/datasets/dialogs-base-08", recipe=DEFAULT_RECIPE, log=print):
    from .dialogs_only import audit_prepared as audit_dialogs
    from .training_run import TrainingOptions, initial_adapter_info, prepare_run_data, primary_examples_per_update
    output, dialogs = Path(output).resolve(), Path(dialogs).resolve()
    config = json.loads(Path(recipe).read_text())
    opts = TrainingOptions.parse(config["options"])
    if (config.get("resume_checkpoint") or not config.get("initial_adapter")
            or config["length_mode"] != "Epochs" or opts.profile != "backbone"
            or opts.sampling_strategy != "coverage_replay" or opts.replay_ratio != .5
            or opts.max_prompt_seconds != 30. or opts.max_target_seconds != 30.
            or opts.validation_scope != "recording" or not opts.validation_all_targets):
        raise ValueError("Require the Dialogs-08 continuation recipe with 30s coverage, equal replay and full validation")
    dialogs_report, dialogs_config, _ = audit_dialogs(dialogs)
    old_train, old_val, _ = prepare_run_data(dialogs_config["manifest_path"], dialogs_config["options"], dialogs_config["seed"])
    adapter = Path(config["initial_adapter"]).resolve()
    identity = initial_adapter_info(adapter, config["rank"], config["alpha"], config["dropout"], opts.profile)
    metadata = json.loads((adapter / "firered_lora_metadata.json").read_text())
    for split in ("train", "validation"):
        if metadata["recipe"].get(split + "_fingerprint") != dialogs_report["datasets"][split]["fingerprint"]:
            raise ValueError("Initial adapter was not trained on the audited Dialogs-08 dataset")
    if metadata.get("initial_adapter") or metadata["recipe"].get("initial_adapter_fingerprint"):
        raise ValueError("Expected the fresh-from-base Dialogs-08 adapter")
    opts.check_base(identity["base_fingerprint"])
    run = Path(config["output_dir"]).resolve()
    for source in (dialogs, adapter, run, Path(archive).resolve()):
        if output == source or output.is_relative_to(source) or source.is_relative_to(output):
            raise ValueError("Preparation output must be separate from source data/adapter/run")
    if run.exists() and any(run.iterdir()):
        raise ValueError("Training output is populated; resume its existing recipe")
    output.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output.parent / f".{output.name}.prepare.lock"), timeout=0):
        if output.exists():
            raise ValueError("Preparation output already exists; use audit or its saved recipe")
        with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
            temporary = Path(temporary)
            source = extract_verified(archive, temporary / "source")
            staging = temporary / "prepared"
            staging.mkdir()
            blocked = {lexical_key(r["text"]) for r in old_train.rows + old_val.rows}
            pool, excluded = collect(source, staging / "audio", blocked)
            train, validation, skipped = select(pool, config["seed"])
            chosen = {r["audio"] for r in train + validation}
            for row in pool:
                if row["audio"] not in chosen:
                    Path(row["audio"]).unlink()
            all_rows = dict(train=train, validation=validation,
                            dialogs_train=old_train.rows, dialogs_validation=old_val.rows)
            for name, filename in MANIFESTS.items():
                write_manifest(staging / filename, all_rows[name])
            config["manifest_path"] = str(staging / MANIFESTS["train"])
            config["options"].update(validation_manifest=str(staging / MANIFESTS["validation"]),
                replay_manifest=str(staging / MANIFESTS["dialogs_train"]),
                adaptation_validation_manifest=str(staging / MANIFESTS["dialogs_validation"]))
            checked = prepare_run_data(config["manifest_path"], config["options"], config["seed"], include_adaptation_validation=True)
            frozen = staging / "initial-adapter"
            frozen.mkdir()
            for name in ADAPTER_FILES:
                shutil.copyfile(adapter / name, frozen / name)
            if initial_adapter_info(frozen, config["rank"], config["alpha"], config["dropout"], opts.profile)["fingerprint"] != identity["fingerprint"]:
                raise ValueError("Dialogs-08 adapter changed during preparation")
            all_rows = {name: pairs.rows for name, pairs in zip(MANIFESTS, checked)}
            for name, rows in all_rows.items():
                if name in {"train", "validation"}:
                    for row in rows:
                        row["audio"] = str(output / "audio" / Path(row["audio"]).name)
                write_manifest(staging / MANIFESTS[name], rows)
            config.update(manifest_path=str(output / MANIFESTS["train"]), output_dir=str(run),
                          initial_adapter=str(output / "initial-adapter"))
            config["options"].update(validation_manifest=str(output / MANIFESTS["validation"]),
                replay_manifest=str(output / MANIFESTS["dialogs_train"]),
                adaptation_validation_manifest=str(output / MANIFESTS["dialogs_validation"]),
                cache_dir=str(Path(opts.cache_dir).resolve()))
            write_json(staging / "recipe.json", config)
            primary = primary_examples_per_update(opts, config["gradient_accumulation"])
            steps = math.ceil(len(train)/primary)
            selected = all_rows["train"] + all_rows["validation"]
            report = dict(schema=SCHEMA, complete=True, dataset_id=DATASET_ID, archive_sha256=ARCHIVE_SHA256,
                initial_adapter_source=str(adapter), initial_adapter_fingerprint=identity["fingerprint"],
                initialization="continue Dialogs-08 LoRA; fresh optimizer/schedule; original base unchanged",
                filters=dict(human_confirmed_only=True, minimum_votes=1, max_audio_seconds=30,
                    max_train_minutes_per_actor=30, max_extra_validation_minutes_per_actor=10/3,
                    validation_fraction_target=.1, thirds_by="record count, separately per actor and split",
                    emotion_filter=None, disfluencies="keep transcribed words, repetitions and internal ellipses"),
                audio_processing="whole recording; resample 24kHz PCM16; fixed attenuation only for overshoot; no cropping/limiter/compressor",
                candidate_pool=summary(pool), datasets={name: summary(rows) for name, rows in all_rows.items()},
                rejections=dict(Counter(r["reason"] for r in excluded)), skipped_actors=skipped,
                annotations_converted_clips=sum(bool(r["cv_transcript_changes"]) for r in selected),
                attenuated_clips=sum(r["cv_audio_processing"]["fixed_gain"] < 1 for r in selected),
                validation="known speakers, distinct recordings and word sequences; original Mozilla dev/test re-partitioned; not an official ASR benchmark",
                primary_validation="Common Voice", secondary_validation="Dialogs-08",
                primary_examples_per_full_update=primary, replay_examples_per_full_update=primary,
                updates_per_epoch=steps, maximum_steps=steps*config["training_epochs"],
                recipe=str(output / "recipe.json"), recipe_sha256=file_digest(staging / "recipe.json"))
            write_json(staging / "filter_audit.json", excluded)
            write_json(staging / "report.json", report)
            staging.rename(output)
    audit(output)
    log(f"[common-voice] Ready: {len(train)} targets; {steps} updates/epoch, up to {report['maximum_steps']} steps.")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("download", help="Download pinned archive; hidden API-key prompt or MOZILLA_DATA_COLLECTIVE_API_KEY")
    fetch.add_argument("--archive", default=DEFAULT_ARCHIVE)
    stage = commands.add_parser("prepare", help="Prepare a new Dialogs-08 continuation without GPU/ASR")
    stage.add_argument("--archive", default=DEFAULT_ARCHIVE)
    stage.add_argument("--output", default=DEFAULT_OUTPUT)
    stage.add_argument("--dialogs", default="training/datasets/dialogs-base-08")
    stage.add_argument("--recipe", default=DEFAULT_RECIPE)
    check = commands.add_parser("audit", help="Recheck prepared data and frozen initial adapter")
    check.add_argument("output", nargs="?", default=DEFAULT_OUTPUT)
    inspect = commands.add_parser("inspect", help="Audit the archive and prospective selection without Dialogs/model files")
    inspect.add_argument("--archive", default=DEFAULT_ARCHIVE)
    args = parser.parse_args()
    if args.command == "download":
        print(download(args.archive))
        return
    if args.command == "prepare":
        result = prepare(args.archive, args.output, args.dialogs, args.recipe)
    elif args.command == "audit":
        result = audit(args.output)
    else:
        with tempfile.TemporaryDirectory(prefix="cv-inspect-") as temporary:
            source = extract_verified(args.archive, Path(temporary) / "source")
            pool, excluded = collect(source, Path(temporary) / "audio")
            train, validation, skipped = select(pool)
            result = dict(candidates=summary(pool), train=summary(train), validation=summary(validation),
                          rejections=dict(Counter(r["reason"] for r in excluded)), skipped_actors=skipped,
                          note="Before Dialogs cross-corpus text exclusion; no training recipe created")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

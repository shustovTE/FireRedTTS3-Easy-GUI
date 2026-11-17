"""Scripted confirmed-only data, global text holdout and unchanged safe audio."""
import csv
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

from fireredtts3_local import common_voice_scripted as cv
from fireredtts3_local import workspace_paths


def metadata(root, rows):
    root.mkdir(parents=True, exist_ok=True)
    values = {
        "validated.tsv": (list(rows[0]), rows),
        "validated_sentences.tsv": (["sentence_id", "sentence"],
                                     [dict(sentence_id=r["sentence_id"], sentence=r["sentence"]) for r in rows]),
        "reported.tsv": (["sentence_id", "reason"], []),
        "clip_durations.tsv": (["clip", "duration[ms]"],
                               [dict(clip=r["path"], **{"duration[ms]": "2000"}) for r in rows]),
    }
    for name, (fields, records) in values.items():
        with (root / name).open("w", encoding="utf-8", newline="") as out:
            writer = csv.DictWriter(out, fields, delimiter="\t")
            writer.writeheader()
            writer.writerows(records)


def records(count=12, speaker="a"):
    return [dict(client_id=speaker, path=f"common_voice_ru_{i+1}.mp3", sentence_id=f"sentence-{i+1}",
                 sentence=f"Это запись номер {i+1}.", up_votes="2", down_votes="0", locale="ru") for i in range(count)]


def planned_rows(count=8, speakers=2, duration=10.):
    return [dict(utt_id=f"cv-scripted27:{s}-{i}", source_id=f"cv-scripted27:{s}-{i}",
                 speaker_id=f"cv-scripted27:{s}", text=f"Общая фраза номер {i}.", duration=duration,
                 category="statement") for s in range(speakers) for i in range(count)]


class TextTests(unittest.TestCase):
    def test_preserves_words_accents_and_internal_disfluencies(self):
        for text in ("Да.", "А вы готовы?!", "— Я, я думаю, что да!", "«Э́то удивительно!»", "„Да!“",
                     "“Это верно.”", "Двадцать плюс два — двадцать два.", "2026 год наступил."):
            self.assertEqual(cv.transcript(text), text)
        self.assertEqual(cv.transcript("  Это\tверно.\n"), "Это верно.")

    def test_rejects_obvious_fragments_without_completing_them(self):
        for text in ("а заработать вообще.", "Она учит нас тому", "Это...", "Это…?", "Это! . .",
                     "«Да.", "Это (верно.", "Это верно).", '"Это верно.', "Двадцать руб.", "", "..."):
            with self.subTest(text=text), self.assertRaises(ValueError):
                cv.transcript(text)

    def test_data_module_does_not_import_models_or_torch(self):
        code = "import sys; import fireredtts3_local.common_voice_scripted; assert not {'torch','torchaudio','fireredtts3_local.training_run'} & set(sys.modules)"
        subprocess.run([sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[1], check=True)


class MetadataTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_requires_both_confirmed_audio_and_exact_confirmed_sentence(self):
        rows = records()
        metadata(self.root, rows)
        # Other/invalidated/ASR-only files are never a candidate source.
        (self.root / "other.tsv").write_text("sentence\nЭто не проверено.\n")
        with (self.root / "reported.tsv").open("a") as out:
            out.write("sentence-1\tgrammar-or-spelling\n")
        path = self.root / "validated_sentences.tsv"
        path.write_text(path.read_text().replace("Это запись номер 2.", "Другие слова.").replace("sentence-3", "missing-3"))
        rows[3]["client_id"] = ""
        rows[4]["sentence"] = "обрывок с точкой."
        rows[5]["path"] = "../../bad.mp3"
        rows[6]["down_votes"] = "2"  # Validated membership, not invented vote cutoff.
        with (self.root / "validated.tsv").open("w", newline="") as out:
            writer = csv.DictWriter(out, rows[0].keys(), delimiter="\t")
            writer.writeheader(); writer.writerows(rows)
        found, excluded, hashes = cv.candidates(self.root)
        reasons = {r["reason"] for r in excluded}
        self.assertTrue({"reported_sentence", "validated_sentence_mismatch", "sentence_not_human_validated",
                         "missing_speaker", "invalid_audio_path"} <= reasons)
        self.assertTrue(any(r["cv_down_votes"] == 2 for r in found))
        self.assertEqual(set(hashes), set(cv.METADATA))
        self.assertTrue(all(r["cv_validated"] for r in found))

    def test_missing_transcript_and_duration_are_reported(self):
        rows = records()
        rows[0]["sentence"] = ""
        metadata(self.root, rows)
        durations = self.root / "clip_durations.tsv"
        durations.write_text(durations.read_text().replace("common_voice_ru_2.mp3\t2000", "common_voice_ru_2.mp3\tnan"))
        _, excluded, _ = cv.candidates(self.root)
        self.assertEqual({r["reason"] for r in excluded}, {"missing_transcript", "missing_or_invalid_duration"})


class SelectionTests(unittest.TestCase):
    def test_shared_prompts_cannot_cross_splits_and_order_is_reproducible(self):
        rows = planned_rows()
        first = cv.select(rows)
        second = cv.select(list(reversed(rows)))
        self.assertEqual([[r["utt_id"] for r in pool] for pool in first[:2]],
                         [[r["utt_id"] for r in pool] for pool in second[:2]])
        train, val = first[:2]
        self.assertFalse({cv.lexical_key(r["text"]) for r in train} & {cv.lexical_key(r["text"]) for r in val})
        for s in {r["speaker_id"] for r in rows}:
            self.assertGreaterEqual(sum(r["speaker_id"] == s for r in train), 2)
            self.assertGreaterEqual(sum(r["speaker_id"] == s for r in val), 2)

    def test_existing_corpus_text_anchors_are_respected(self):
        rows = planned_rows()
        existing_train, existing_val = (cv.lexical_key(rows[i]["text"]) for i in (0, 1))
        train, val, _, _ = cv.select(rows, blocked_train_texts=[existing_val], blocked_validation_texts=[existing_train])
        self.assertNotIn(existing_val, {cv.lexical_key(r["text"]) for r in train})
        self.assertNotIn(existing_train, {cv.lexical_key(r["text"]) for r in val})
        with self.assertRaisesRegex(ValueError, "already leak"):
            cv.select(rows, blocked_train_texts=[existing_val], blocked_validation_texts=[existing_val])

    def test_no_clip_ceiling_and_long_whole_validation_pair_is_explicit(self):
        rows = planned_rows(4, 1, 150.)
        train, val, _, exceptions = cv.select(rows)
        self.assertEqual((len(train), len(val)), (2, 2))
        self.assertEqual(sum(r["duration"] for r in val), 300.)
        self.assertEqual(exceptions[0]["seconds"], 300.)
        self.assertTrue(all(r["duration"] > 30 for r in train + val))

    def test_validation_is_extra_to_training_budget(self):
        train, val, _, _ = cv.select(planned_rows(12, 1, 100.), train_minutes=4.)
        self.assertLessEqual(sum(r["duration"] for r in train), 240.)
        self.assertGreater(sum(r["duration"] for r in train + val), 240.)
        self.assertEqual(len(val), 2)

    def test_same_speaker_same_words_cannot_be_independent_references(self):
        rows = planned_rows(4, 1)
        rows[3]["text"] = rows[2]["text"]
        with self.assertRaisesRegex(ValueError, "No speakers"):
            cv.select(rows)


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def archive(self):
        source = self.root / "source"
        rows = records()
        metadata(source, rows)
        (source / "clips").mkdir()
        for i, row in enumerate(rows):
            wave = .1*np.sin(np.arange(48000, dtype=np.float32)*(.02+i*.001))
            wave[-1] = .03
            sf.write(source / "clips" / row["path"], wave, 24000, format="WAV", subtype="FLOAT")
        (source / "clips" / "not_validated.mp3").write_bytes(b"must never be selected")
        archive = self.root / "corpus.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            for name in cv.METADATA:
                tar.add(source / name, arcname=f"{cv.CORPUS}/{name}")
            for path in (source / "clips").iterdir():
                tar.add(path, arcname=f"{cv.CORPUS}/clips/{path.name}")
        for key, value in (("ARCHIVE_BYTES", archive.stat().st_size), ("ARCHIVE_SHA256", cv.file_digest(archive))):
            p = patch.object(cv, key, value)
            p.start(); self.addCleanup(p.stop)
        return archive, source

    def test_prepare_keeps_exact_original_audio_and_audit_detects_changes(self):
        archive, source = self.archive()
        output = self.root / "prepared"
        report = cv.prepare(archive, output, log=lambda _: None)
        self.assertTrue(report["complete"])
        self.assertIsNone(report["max_clip_seconds"])
        self.assertEqual(cv.audit(output), report)
        self.assertFalse((output / "audio" / "not_validated.mp3").exists())
        rows = [json.loads(line) for line in (output / "train.jsonl").read_text().splitlines()]
        for row in rows:
            self.assertEqual(Path(row["audio"]).read_bytes(), (source / "clips" / row["cv_source_path"]).read_bytes())
            self.assertEqual(sf.info(row["audio"]).frames, 48000)
        Path(rows[0]["audio"]).write_bytes(b"modified")
        with self.assertRaisesRegex(ValueError, "audio changed"):
            cv.audit(output)

    def test_selective_extraction_rejects_links_and_traversal(self):
        for i, (name, link) in enumerate(((f"{cv.CORPUS}/../../escape", False),
                                         (f"{cv.CORPUS}/validated.tsv", True), ("/absolute", False))):
            archive = self.root / f"bad-{i}.tar.gz"
            with tarfile.open(archive, "w:gz") as tar:
                member = tarfile.TarInfo(name)
                if link:
                    member.type = tarfile.SYMTYPE; member.linkname = "/etc/passwd"
                tar.addfile(member)
            with self.assertRaisesRegex(ValueError, "Unsafe"):
                cv.extract_metadata(archive, self.root / f"out-{i}")
        self.assertFalse((self.root / "escape").exists())

    def test_workspace_move_without_alias_preserves_scripted_manifest_and_content_audit(self):
        archive, _ = self.archive()
        output = self.root / "prepared"
        report = cv.prepare(archive, output, log=lambda _: None)
        before = {name: (output / name).read_bytes()
                  for name in ("report.json", "train.jsonl", "validation.jsonl")}
        with tempfile.TemporaryDirectory() as destination:
            moved = Path(destination) / "project"
            self.root.rename(moved)
            try:
                with patch.multiple(workspace_paths, LEGACY_ROOT=self.root, WORKSPACE_ROOT=moved):
                    new_output = moved / "prepared"
                    self.assertFalse(self.root.exists())
                    self.assertFalse(self.root.is_symlink())
                    self.assertEqual(cv.audit(new_output), report)
                    self.assertEqual({name: (new_output / name).read_bytes() for name in before}, before)
                    row = json.loads((new_output / "train.jsonl").read_text().splitlines()[0])
                    workspace_paths.local_path(row["audio"]).write_bytes(b"modified")
                    with self.assertRaisesRegex(ValueError, "audio changed"):
                        cv.audit(new_output)
                    self.assertFalse(self.root.exists())
            finally:
                moved.rename(self.root)


class Response(io.BytesIO):
    def __init__(self, data, status=200, headers=None):
        super().__init__(data)
        self.status = status
        self.headers = headers or {}


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.payload = b"complete archive fixture"
        for name, value in (("ARCHIVE_BYTES", len(self.payload)), ("ARCHIVE_SHA256", hashlib.sha256(self.payload).hexdigest())):
            p = patch.object(cv, name, value)
            p.start(); self.addCleanup(p.stop)

    def test_resumes_pinned_partial_without_sending_api_key_to_storage(self):
        archive = self.root / "archive.tar.gz"
        archive.with_name(archive.name + ".part").write_bytes(self.payload[:8])
        response = Response(self.payload[8:], 206, {"Content-Range": f"bytes 8-{len(self.payload)-1}/{len(self.payload)}"})
        logs = []
        with patch.dict("os.environ", MOZILLA_DATA_COLLECTIVE_API_KEY="fixture-api-secret"), \
             patch.object(cv.urllib.request, "build_opener") as opener, \
             patch.object(cv.urllib.request, "urlopen", return_value=response) as storage:
            opener.return_value.open.return_value = Response(b'{"downloadUrl":"https://storage.example/data?secret-signature"}')
            cv.download(archive, log=logs.append)
            self.assertEqual(opener.return_value.open.call_args.args[0].get_header("Authorization"), "Bearer fixture-api-secret")
            self.assertEqual(opener.return_value.open.call_args.args[0].get_header("User-agent"), cv.USER_AGENT)
            request = storage.call_args.args[0]
            self.assertIsNone(request.get_header("Authorization"))
            self.assertEqual(request.get_header("User-agent"), cv.USER_AGENT)
            self.assertEqual(request.get_header("Range"), "bytes=8-")
            cv.download(archive, log=logs.append)
            self.assertEqual(storage.call_count, 1)
        self.assertEqual(archive.read_bytes(), self.payload)
        self.assertNotIn("secret", "\n".join(logs))
        self.assertFalse(archive.with_name(archive.name + ".part").exists())

    def test_server_ignoring_range_replaces_partial_instead_of_appending(self):
        archive = self.root / "archive.tar.gz"
        archive.with_name(archive.name + ".part").write_bytes(self.payload[:8])
        with patch.object(cv, "_download_url", return_value="https://storage.example/private"), \
             patch.object(cv.urllib.request, "urlopen", return_value=Response(self.payload)):
            cv.download(archive, log=lambda _: None)
        self.assertEqual(archive.read_bytes(), self.payload)

    def test_transport_failure_retains_partial_but_redacts_signed_url(self):
        archive = self.root / "archive.tar.gz"
        partial = archive.with_name(archive.name + ".part")
        partial.write_bytes(self.payload[:8])
        with patch.object(cv, "_download_url", return_value="https://storage.example/private-signature"), \
             patch.object(cv.urllib.request, "urlopen", side_effect=TimeoutError("private-signature")):
            with self.assertRaises(RuntimeError) as caught:
                cv.download(archive, log=lambda _: None)
        self.assertNotIn("private-signature", str(caught.exception))
        self.assertEqual(partial.read_bytes(), self.payload[:8])
        self.assertFalse(archive.exists())

    def test_wrong_content_range_cannot_corrupt_existing_partial(self):
        archive = self.root / "archive.tar.gz"
        partial = archive.with_name(archive.name + ".part")
        partial.write_bytes(self.payload[:8])
        with patch.object(cv, "_download_url", return_value="https://storage.example/private"), \
             patch.object(cv.urllib.request, "urlopen", return_value=Response(self.payload[8:], 206, {"Content-Range": "bytes 0-4/5"})):
            with self.assertRaises(RuntimeError):
                cv.download(archive, log=lambda _: None)
        self.assertEqual(partial.read_bytes(), self.payload[:8])


if __name__ == "__main__":
    unittest.main()

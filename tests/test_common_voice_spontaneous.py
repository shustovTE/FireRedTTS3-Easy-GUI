"""Confirmed-only selection, disfluency fidelity, whole audio and pinned continuation."""
import csv
import hashlib
import io
import json
from pathlib import Path
import socket
import ssl
import tarfile
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

from fireredtts3_local import common_voice_download as fetch
from fireredtts3_local.common_voice_spontaneous import (audit, collect, prepare, select,
                                                       transcript, validate_selection)
from fireredtts3_local.training_data import dataset_fingerprint, file_digest, load_manifest, write_manifest
from fireredtts3_local.training_run import prepare_run_data

REPO = Path(__file__).resolve().parents[1]


class TextTests(unittest.TestCase):
    def test_keeps_spoken_disfluencies_and_maps_only_known_labels(self):
        text = "Эм, ну... А-а, я, я думаю, да."
        self.assertEqual(transcript(text), (text, []))
        actual, changes = transcript("[pause] Ну, [disfluency|эм], [pause] пожалуй, да.[silence]")
        self.assertEqual(actual, "Ну, эм, … пожалуй, да.")
        self.assertEqual(len(changes), 4)
        for text in ("Вело[pause]дорожки.", "Это [unclear].", "[laugh] Да.",
                     "Это [disfluency].", "Да [unknown].", "Да [pause.",
                     "Ну…", "Ну...", "Да, maybe.", "[disfluency|unknown] Да."):
            with self.subTest(text=text), self.assertRaises(ValueError):
                transcript(text)


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def pin(self, data):
        for name, value in (("ARCHIVE_BYTES", len(data)), ("ARCHIVE_SHA256", hashlib.sha256(data).hexdigest())):
            p = patch.object(fetch, name, value)
            p.start(); self.addCleanup(p.stop)

    def test_download_separates_bearer_from_storage_and_reuses_verified_archive(self):
        data = b"pinned archive fixture"
        self.pin(data)
        storage_url = "https://storage.example/archive?private-signature=fixture"
        out = self.root / "corpus.tar.gz"
        logs = []
        with patch.dict("os.environ", MOZILLA_DATA_COLLECTIVE_API_KEY="fixture-secret"), \
             patch.object(fetch.urllib.request, "build_opener") as opener, \
             patch.object(fetch.urllib.request, "urlopen", return_value=io.BytesIO(data)) as storage:
            opener.return_value.open.return_value = io.BytesIO(json.dumps({"downloadUrl": storage_url}).encode())
            fetch.download(out, log=logs.append)
            request = opener.return_value.open.call_args.args[0]
            self.assertEqual(request.get_header("Authorization"), "Bearer fixture-secret")
            self.assertEqual(request.get_header("User-agent"), fetch.USER_AGENT)
            self.assertEqual(request.get_header("Accept"), "application/json")
            storage_request = storage.call_args.args[0]
            self.assertEqual(storage_request.full_url, storage_url)
            self.assertEqual(storage_request.get_header("User-agent"), fetch.USER_AGENT)
            self.assertIsNone(storage_request.get_header("Authorization"))
            fetch.download(out, log=logs.append)
            self.assertEqual(storage.call_count, 1)
        self.assertNotIn("fixture-secret", "\n".join(logs))
        self.assertNotIn("private-signature", "\n".join(logs))
        self.assertEqual(out.read_bytes(), data)
        out.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "differs"):
            fetch.verify_archive(out)

    def test_failed_download_sanitizes_signed_url_and_installs_nothing(self):
        out = self.root / "corpus.tar.gz"
        with patch.dict("os.environ", MOZILLA_DATA_COLLECTIVE_API_KEY="fixture-secret"), \
             patch.object(fetch.urllib.request, "build_opener") as opener, \
             patch.object(fetch.urllib.request, "urlopen", side_effect=RuntimeError("private signed URL")):
            opener.return_value.open.return_value = io.BytesIO(b'{"downloadUrl":"https://storage.example/private"}')
            with self.assertRaises(RuntimeError) as caught:
                fetch.download(out)
        self.assertNotIn("private", str(caught.exception))
        self.assertFalse(out.exists())
        self.assertIsNone(fetch.NoRedirect().redirect_request(None, None, 302, "", {}, "https://elsewhere"))

    def test_http_errors_distinguish_cloudflare_from_auth_without_reflecting_secrets(self):
        for status, body, expected in (
            (403, {"cloudflare_error": True, "error_code": 1010}, "Cloudflare 1010"),
            (401, {}, "Authentication was rejected"),
            (403, {}, "not necessarily an invalid key"),
            (429, {}, "Rate limited"),
            (503, {}, "5xx"),
        ):
            body.update(error="fixture-secret", downloadUrl="https://storage.example/?secret-signature")
            error = fetch.urllib.error.HTTPError("https://api.example/?secret-signature", status,
                "fixture-secret", {"Content-Type": "application/json"}, io.BytesIO(json.dumps(body).encode()))
            result = str(fetch.safe_failure("Mozilla download API", error))
            self.assertIn(f"HTTP {status}", result)
            self.assertIn(expected, result)
            self.assertNotIn("fixture-secret", result)
            self.assertNotIn("secret-signature", result)

    def test_network_and_invalid_json_errors_are_actionable_without_raw_exception_text(self):
        for error, expected in (
            (fetch.urllib.error.URLError(ssl.SSLCertVerificationError("secret-signature")), "TLS certificate"),
            (fetch.urllib.error.URLError(socket.gaierror("secret-signature")), "DNS lookup"),
            (TimeoutError("secret-signature"), "timed out"),
            (json.JSONDecodeError("secret-signature", "", 0), "non-JSON"),
        ):
            result = str(fetch.safe_failure("Mozilla download API", error))
            self.assertIn(expected, result)
            self.assertNotIn("secret-signature", result)

    def test_challenge_header_is_distinguished_without_exposing_html_or_headers(self):
        for header in ("cf-mitigated", "CF-Mitigated"):
            error = fetch.urllib.error.HTTPError("https://api.example/?secret-signature", 403,
                "fixture-secret", {"Content-Type": "text/html; charset=UTF-8", header: "challenge",
                                   "X-Debug": "fixture-secret"},
                io.BytesIO(b"<html>private challenge fixture-secret secret-signature</html>"))
            result = str(fetch.safe_failure("Mozilla download API", error))
            self.assertIn("HTTP 403, response=HTML", result)
            self.assertIn("cf-mitigated: challenge", result)
            self.assertIn("not an API-key validation result", result)
            self.assertNotIn("fixture-secret", result)
            self.assertNotIn("secret-signature", result)

    def test_html_403_without_challenge_header_does_not_guess_its_cause(self):
        error = fetch.urllib.error.HTTPError("https://api.example", 403, "Forbidden",
            {"Content-Type": "text/html"}, io.BytesIO(b"<html>Forbidden</html>"))
        result = str(fetch.safe_failure("Mozilla download API", error))
        self.assertIn("not necessarily an invalid key", result)
        self.assertNotIn("Cloudflare challenge", result)

    def test_api_failure_is_not_relabelled_as_invalid_key_and_does_not_start_storage(self):
        error = fetch.urllib.error.HTTPError("https://api.example", 403, "Forbidden",
            {"Content-Type": "application/json"}, io.BytesIO(b'{"cloudflare_error":true,"error_code":1010}'))
        with patch.dict("os.environ", MOZILLA_DATA_COLLECTIVE_API_KEY="fixture-secret"), \
             patch.object(fetch.urllib.request, "build_opener") as opener, \
             patch.object(fetch.urllib.request, "urlopen") as storage:
            opener.return_value.open.side_effect = error
            with self.assertRaisesRegex(RuntimeError, "HTTP 403.*Cloudflare 1010"):
                fetch.download(self.root / "not-created.tar.gz")
            storage.assert_not_called()
        self.assertFalse((self.root / "not-created.tar.gz").exists())

    def test_extraction_rejects_links_traversal_and_wrong_root(self):
        for name, symlink in ((fetch.CORPUS + "/../../escape", False), ("/absolute", False),
                              ("wrong-root/file", False), (fetch.CORPUS + "/link", True)):
            archive = self.root / "bad.tar.gz"
            with tarfile.open(archive, "w:gz") as tar:
                entry = tarfile.TarInfo(name)
                if symlink:
                    entry.type = tarfile.SYMTYPE; entry.linkname = "/etc/passwd"
                tar.addfile(entry)
            with patch.object(fetch, "verify_archive"), self.assertRaisesRegex(ValueError, "Unsafe"):
                fetch.extract_verified(archive, self.root / "unpacked")
        self.assertFalse((self.root / "escape").exists())


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / fetch.CORPUS
        (self.source / "audios").mkdir(parents=True)
        self.records = []
        for speaker in ("a"*32, "b"*32):
            for i in range(24):
                number = len(self.records) + 1
                text = (f"Эм, что, что думаешь о номере {number}?" if i%3 == 0 else
                        f"Ну... пожалуй, номер {number}." if i%3 == 1 else f"А-а, номер {number}!")
                # Real decoded samples, WAV container under an MP3 name: loader
                # detects its actual format. Production archive is pinned MP3.
                wave = .1*np.sin(np.arange(72000, dtype=np.float32)*(.02+number*.001))
                wave[-1] = .05
                if number == 1:
                    wave[35000] = 1.4  # sparse overshoot, not widespread clipping
                name = f"spontaneous-speech-ru-{number}.mp3"
                sf.write(self.source / "audios" / name, wave, 24000, format="WAV", subtype="FLOAT")
                self.records.append(dict(audio_id=str(number), audio_file=name, client_id=speaker,
                    duration_ms="3000", language="Russian", transcription=text, votes="1",
                    split="test" if speaker[0] == "a" else "dev", quality_tags=""))
        self.write_metadata()
        self.dialogs = self.root / "dialogs"
        self.dialogs.mkdir()
        self.dialogs_config = json.loads((REPO / "recipes/russian_dialogs_base_96gb.json").read_text())
        fingerprints = {}
        for split in ("train", "validation"):
            rows = []
            for actor in "DMS":
                for i in range(3):
                    path = self.dialogs / f"{split}-{actor}-{i}.wav"
                    wave = np.sin(np.arange(60000)*(.1+i*.02+ord(actor)*.001+(split=="train")*.3))*.1
                    sf.write(path, wave, 24000)
                    rows.append(dict(audio=str(path), text=f"Dialogs {split} {actor} номер {i}.",
                        speaker_id="dialogs:"+actor, language="Russian", source_id=f"{split}-{actor}-{i}"))
            manifest = write_manifest(self.dialogs / f"{split}.jsonl", rows)
            fingerprints[split] = dataset_fingerprint(load_manifest(manifest))
        self.dialogs_config["manifest_path"] = str(self.dialogs / "train.jsonl")
        self.dialogs_config["options"]["validation_manifest"] = str(self.dialogs / "validation.jsonl")
        (self.dialogs / "recipe.json").write_text(json.dumps(self.dialogs_config))
        (self.dialogs / "report.json").write_text(json.dumps(dict(schema="dialogs-original-base-v1", complete=True,
            recipe_sha256=file_digest(self.dialogs / "recipe.json"),
            datasets={k: {"fingerprint": v} for k,v in fingerprints.items()})))
        self.adapter = self.root / "dialogs-08-best"
        self.adapter.mkdir()
        (self.adapter / "adapter_model.safetensors").write_bytes(b"fixture weights, not used for GPU training")
        (self.adapter / "adapter_config.json").write_text(json.dumps(dict(peft_type="LORA", r=16,
            lora_alpha=32, lora_dropout=.05, bias="none", target_modules=["backbone_llm.q_proj"])))
        (self.adapter / "firered_lora_metadata.json").write_text(json.dumps(dict(
            base_fingerprint=self.dialogs_config["options"]["expected_base_fingerprint"],
            recipe=dict(train_fingerprint=fingerprints["train"], validation_fingerprint=fingerprints["validation"],
                        options={"profile":"backbone"}))))
        config = json.loads((REPO / "recipes/russian_spontaneous_96gb.json").read_text())
        config["initial_adapter"] = str(self.adapter)
        config["output_dir"] = str(self.root / "run")
        config["options"]["cache_dir"] = str(self.root / "cache")
        self.recipe = self.root / "recipe.json"
        self.recipe.write_text(json.dumps(config))
        self.output = self.root / "prepared"

    def write_metadata(self):
        with (self.source / "ss-corpus-ru.tsv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(self.records[0]), delimiter="\t")
            writer.writeheader(); writer.writerows(self.records)
        (self.source / "ss-reported-audios-ru.tsv").write_text("audio_id\n")

    def archive(self):
        path = self.root / "corpus.tar.gz"
        with tarfile.open(path, "w:gz") as tar:
            tar.add(self.source, arcname=fetch.CORPUS)
        for name,value in (("ARCHIVE_BYTES",path.stat().st_size),("ARCHIVE_SHA256",file_digest(path))):
            p=patch.object(fetch,name,value); p.start(); self.addCleanup(p.stop)
        return path

    def test_confirmation_not_prompt_transcript_and_whole_audio_with_headroom(self):
        self.records[0]["votes"] = "2"
        self.records[0]["prompt"] = "Этот вопрос не произнесён?"
        for r in self.records[1:]:r["prompt"] = "Этот вопрос не произнесён?"
        self.records[1]["votes"] = "0"
        self.records[2]["split"] = ""
        self.records[3]["duration_ms"] = "30001"
        self.records[4]["transcription"] = "Это [unclear]."
        self.write_metadata()
        rows, rejected = collect(self.source, self.root / "audio")
        reasons = [r["reason"] for r in rejected]
        self.assertEqual(reasons.count("not_human_confirmed"), 2)
        self.assertIn("duration", reasons)
        self.assertIn("unsupported_or_unclear_annotation", reasons)
        first = next(r for r in rows if r["cv_audio_id"] == "1")
        self.assertEqual(first["text"], self.records[0]["transcription"])
        source,rate = sf.read(self.source / "audios" / self.records[0]["audio_file"])
        output,newrate = sf.read(first["audio"])
        self.assertEqual((len(source),rate),(len(output),newrate))
        np.testing.assert_allclose(output, source*first["cv_audio_processing"]["fixed_gain"], atol=1/32768)
        self.assertLessEqual(np.abs(output).max(), .981)
        self.assertNotEqual(output[-1], 0.)

    def test_exact_thirds_extra_holdout_and_deterministic_pairs(self):
        rows,_ = collect(self.source, self.root / "audio")
        train,val,skipped = select(rows)
        self.assertEqual((len(train),len(val),len(skipped)),(42,6,0))
        self.assertEqual((train,val,skipped),select(rows[::-1]))
        validate_selection(train,val)
        broken = [dict(r) for r in val]
        broken[0]["text"] = train[0]["text"]
        with self.assertRaisesRegex(ValueError,"leakage"):
            validate_selection(train,broken)

    def test_full_30_second_audio_survives_resampling_and_dialogs_text_is_excluded(self):
        record = self.records[0]
        record["duration_ms"] = "30000"
        wave = (.15*np.sin(np.arange(960000)*.031)).astype(np.float32)
        wave[-3200:] = .05  # preserve the last 100ms, including the final sample
        sf.write(self.source / "audios" / record["audio_file"], wave, 32000, format="WAV", subtype="FLOAT")
        self.write_metadata()
        from fireredtts3_local.dialogs_dataset import lexical_key
        rows,excluded = collect(self.source,self.root / "audio",{lexical_key(self.records[1]["transcription"])})
        first = next(r for r in rows if r["cv_audio_id"] == "1")
        actual,sr = sf.read(first["audio"])
        self.assertEqual((len(actual),sr,first["duration"]),(720000,24000,30.))
        self.assertGreater(actual[-1],.01)
        self.assertEqual(first["start"],0.)
        self.assertTrue(any(r["reason"]=="dialogs_text_overlap" for r in excluded))

    def test_large_pool_respects_separate_train_and_extra_validation_budgets(self):
        rows = []
        for i in range(300):
            kind = ("question","statement","other")[i%3]
            rows.append(dict(audio="unused", audio_sha256=str(i), utt_id=str(i), source_id=str(i),
                text=f"Реплика {i}"+{"question":"?","statement":".","other":"!"}[kind],
                speaker_id="same",language="Russian",duration=30.,category=kind))
        train,val,_ = select(rows)
        self.assertEqual(sum(r["duration"] for r in train),1800.)
        self.assertEqual(sum(r["duration"] for r in val),180.)
        self.assertEqual(len(train),60)
        self.assertEqual(len(val),6)
        validate_selection(train,val)

    def test_prepare_pins_dialogs08_both_holdouts_and_detects_tampering(self):
        archive = self.archive()
        report = prepare(archive,self.output,self.dialogs,self.recipe,log=lambda _:None)
        self.assertEqual(report,audit(self.output))
        config = json.loads((self.output / "recipe.json").read_text())
        self.assertIsNone(config["resume_checkpoint"])
        self.assertEqual(Path(config["initial_adapter"]),self.output / "initial-adapter")
        data = prepare_run_data(config["manifest_path"],config["options"],config["seed"],include_adaptation_validation=True)
        self.assertEqual([len(d.rows) for d in data],[42,6,9,9])
        self.assertEqual(report["maximum_steps"],55)
        # Changing the historical source adapter cannot alter the frozen input.
        (self.adapter / "adapter_model.safetensors").write_bytes(b"later change")
        audit(self.output)
        with self.assertRaises(ValueError):prepare(archive,self.output,self.dialogs,self.recipe)
        Path(data[0].rows[0]["audio"]).write_bytes(Path(data[0].rows[1]["audio"]).read_bytes())
        with self.assertRaisesRegex(ValueError,"changed"):
            audit(self.output)

    def test_wrong_initial_stage_rejected_before_preparation(self):
        path = self.adapter / "firered_lora_metadata.json"
        meta = json.loads(path.read_text())
        meta["recipe"]["train_fingerprint"] = "different corpus"
        path.write_text(json.dumps(meta))
        with self.assertRaisesRegex(ValueError,"Dialogs-08"):
            prepare(self.root / "missing.tar.gz",self.output,self.dialogs,self.recipe)
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()

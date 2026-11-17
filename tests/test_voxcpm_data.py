"""Three-corpus preparation must preserve holdouts, originals and complete audio."""
import json
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

from fireredtts3_local import voxcpm_data as data
from fireredtts3_local import workspace_paths
from fireredtts3_local.training_data import dataset_fingerprint, file_digest


class Detector:
    metadata = {"package": "silero-vad", "version": data.VAD_VERSION, "backend": "torchscript",
                "device": "cpu", "model_resource": "silero_vad.data/silero_vad.jit",
                "model_sha256": data.VAD_MODEL_SHA256, "pypi_wheel_sha256": data.VAD_WHEEL_SHA256,
                "settings": data.VAD_SETTINGS, "torch_version": "test-double"}

    def __call__(self, samples):
        # A contiguous range leaves an internal pause untouched. The production
        # wrapper supplies its 200ms padding and preserves original sample rate.
        return [{"start": int(len(samples) * .3), "end": int(len(samples) * .7)}]


class VoxCPMDataTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.paths, self.rows = {}, {}
        serial = 0
        for corpus, schema in data.SOURCE_SCHEMAS.items():
            folder = self.root / corpus
            folder.mkdir()
            self.paths[corpus] = folder
            self.rows[corpus] = {}
            for split in data.SPLITS:
                rows = []
                for index in range(4):
                    serial += 1
                    path = folder / f"{split}-{index}.wav"
                    samples = .15 * np.sin(np.arange(16000) * (serial + 1) / 37)
                    samples[7500:8500] = 0  # Internal pause must survive.
                    sf.write(path, samples, 16000, subtype="FLOAT")
                    rows.append({"audio": str(path), "text": f"Реплика {corpus} {split} номер {index}.",
                        "speaker_id": corpus + ":speaker", "utt_id": f"{corpus}:{split}:{index}",
                        "source_id": f"{corpus}:recording:{split}:{index}", "duration": 1.,
                        "start": 0., "end": 1., "language": "Russian", "audio_sha256": file_digest(path),
                        "category": "statement", "selection_slot": "random", "selection_triple": index})
                self.rows[corpus][split] = rows
            self.save(corpus)
        self.output = self.root / "prepared"

    def save(self, corpus):
        folder = self.paths[corpus]
        for split, rows in self.rows[corpus].items():
            (folder / f"{split}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        (folder / "report.json").write_text(json.dumps({"schema": data.SOURCE_SCHEMAS[corpus],
            "complete": True, "datasets": {split: {"fingerprint": dataset_fingerprint(rows)}
                                           for split, rows in self.rows[corpus].items()}}))

    def prepare(self, **kwargs):
        return data.prepare(**self.paths, output=self.output, detector=kwargs.pop("detector", Detector()),
                            log=lambda _: None, **kwargs)

    def provenance(self, split):
        return [json.loads(line) for line in (self.output / f"{split}-provenance.jsonl").read_text().splitlines()]

    def test_all_three_corpora_trimmed_originals_preserved_and_pairs_audited(self):
        originals = {row["audio"]: Path(row["audio"]).read_bytes() for splits in self.rows.values()
                     for rows in splits.values() for row in rows}
        report = self.prepare()
        self.assertEqual(report["source_clips"], 24)
        self.assertEqual(report["vad_successful_clips"], 24)
        self.assertEqual(report["rejections"], {})
        self.assertEqual(data.audit(self.output), report)
        for split in data.SPLITS:
            provenance = self.provenance(split)
            self.assertEqual({row["dataset_id"] for row in provenance}, set(data.SOURCE_SCHEMAS))
            by_audio = {row["audio"]: row for row in provenance}
            for row in provenance:
                self.assertAlmostEqual(row["duration"], .8)
                ref = by_audio[row["ref_audio"]]
                self.assertEqual(ref["speaker_id"], row["speaker_id"])
                self.assertEqual(ref["dataset_id"], row["dataset_id"])
                self.assertNotEqual(ref["text"], row["text"])
                self.assertNotEqual(ref["audio_sha256"], row["audio_sha256"])
                source, rate = sf.read(row["source_audio"], dtype="float32")
                trimmed, trimmed_rate = sf.read(row["audio"], dtype="float32")
                self.assertEqual(trimmed_rate, rate)
                np.testing.assert_array_equal(trimmed, source[1600:14400])
                self.assertEqual(row["selection_slot"], "random")
        for path, before in originals.items():
            self.assertEqual(Path(path).read_bytes(), before)
        with patch.object(data, "_trim_audio", side_effect=AssertionError("must not repeat VAD")):
            self.assertEqual(self.prepare(), report)

    def test_uncapped_sources_filter_after_vad_and_only_pair_within_limit(self):
        class Edges(Detector):
            def __call__(self, samples):
                # Remove exactly one second total after 200 ms of padding on both edges.
                return [{"start": 11200, "end": len(samples) - 11200}] if len(samples) > 32000 else super().__call__(samples)

        for corpus, schema in data.RAW_SOURCES.items():
            for split, rows in self.rows[corpus].items():
                for i, row in enumerate(rows):
                    row.update({"emotion": "neutral"} if corpus == "dialogs" else {"cv_validated": True})
                    row["text"] = row["text"].removesuffix(".") + ("?" if i % 2 == 0 else ".")
                    if i != 3:
                        frames = 31 * 16000 + i if i < 2 else 7 * 16000
                        path = Path(row["audio"])
                        samples = np.sin(np.arange(frames) * (.02 + (.03 if corpus == "dialogs" else .04)
                                                               + (.005 if split == "train" else 0) + i * .001))
                        sf.write(path, samples, 16000, subtype="FLOAT")
                        row.update(audio_sha256=file_digest(path), duration=frames / 16000)
            self.save(corpus)
            path = self.paths[corpus] / "report.json"
            path.write_text(json.dumps(dict(json.loads(path.read_text()), schema=schema)))
        report = data.prepare_raw_30s(self.root, self.output, detector=Edges(), log=lambda _: None)
        self.assertEqual(report, data.audit(self.output))
        self.assertEqual(report["rejections"], {"after_vad_over_30s": 4, "train_after_vad_under_3s": 2})
        self.assertEqual(set(report["sources"]), {"dialogs", "scripted"})
        for split in data.SPLITS:
            rows = self.provenance(split)
            self.assertEqual(len(rows), 4 if split == "train" else 6)
            self.assertEqual(max(r["duration"] for r in rows), 30.)
            self.assertLessEqual(max(r["ref_duration"] for r in rows), 30.)
            self.assertFalse(any(r["utt_id"].endswith(":1") for r in rows))
            self.assertFalse(any(r["ref_utt_id"].endswith(":1") for r in rows))

    def test_cross_corpus_normalized_text_holdout_wins(self):
        self.rows["dialogs"]["train"][0]["text"] = "Всё готово?"
        self.rows["scripted"]["validation"][0]["text"] = "ВСЕ готово."
        self.save("dialogs"); self.save("scripted")
        report = self.prepare()
        self.assertEqual(report["rejections"]["holdout_overlap"], 1)
        self.assertNotIn("dialogs:train:0", {row["utt_id"] for row in self.provenance("train")})
        self.assertIn("scripted:validation:0", {row["utt_id"] for row in self.provenance("validation")})
        self.assertEqual(report["corpora"]["dialogs"]["train"]["source"]["clips"], 4)
        self.assertEqual(report["corpora"]["dialogs"]["train"]["retained"]["clips"], 3)

    def test_workspace_move_without_alias_reuses_data_without_rewriting_identity(self):
        report = self.prepare()
        before = {path.relative_to(self.root): path.read_bytes()
                  for path in self.root.rglob("*.json*")}
        with tempfile.TemporaryDirectory() as destination:
            moved = Path(destination) / "project"
            self.root.rename(moved)
            try:
                with patch.multiple(workspace_paths, LEGACY_ROOT=self.root, WORKSPACE_ROOT=moved):
                    self.assertFalse(self.root.exists())
                    self.assertFalse(self.root.is_symlink())
                    new_paths = {name: moved / path.relative_to(self.root) for name, path in self.paths.items()}
                    new_output = moved / self.output.relative_to(self.root)
                    self.assertEqual(data.audit(new_output), report)
                    with patch.object(data, "_trim_audio", side_effect=AssertionError("must not repeat VAD")):
                        self.assertEqual(data.prepare(**new_paths, output=new_output, log=lambda _: None), report)
                    self.assertEqual({path: (moved / path).read_bytes() for path in before}, before)
                    # Relocation never replaces content/provenance checks.
                    source = workspace_paths.local_path(self.rows["dialogs"]["train"][0]["audio"])
                    samples, rate = sf.read(source)
                    sf.write(source, samples * .5, rate, subtype="FLOAT")
                    with self.assertRaisesRegex(ValueError, "Source audio hash changed"):
                        data.audit(new_output)
                    self.assertFalse(self.root.exists())
            finally:
                moved.rename(self.root)

    def test_fresh_preparation_after_move_uses_new_canonical_paths(self):
        original_manifest = (self.paths["dialogs"] / "train.jsonl").read_bytes()
        with tempfile.TemporaryDirectory() as destination:
            moved = Path(destination) / "project"
            self.root.rename(moved)
            try:
                with patch.multiple(workspace_paths, LEGACY_ROOT=self.root, WORKSPACE_ROOT=moved):
                    new_paths = {name: moved / path.relative_to(self.root) for name, path in self.paths.items()}
                    new_output = moved / "prepared"
                    report = data.prepare(**new_paths, output=new_output, detector=Detector(), log=lambda _: None)
                    for spec in report["sources"].values():
                        self.assertTrue(Path(spec["path"]).is_relative_to(moved))
                    for row in data._read_rows(new_output / "train-provenance.jsonl"):
                        self.assertTrue(Path(row["audio"]).is_relative_to(moved))
                        self.assertTrue(Path(row["source_audio"]).is_relative_to(moved))
                    self.assertEqual(data.audit(new_output), report)
                    self.assertEqual((new_paths["dialogs"] / "train.jsonl").read_bytes(), original_manifest)
                    self.assertFalse(self.root.exists())
            finally:
                moved.rename(self.root)

    def test_validation_interleaves_corpora_and_speakers_without_losing_targets(self):
        expected = set()
        for corpus in self.paths:
            for index, row in enumerate(self.rows[corpus]["validation"]):
                row["speaker_id"] = f"{corpus}:speaker-{index // 2}"
                expected.add(row["utt_id"])
            self.save(corpus)
        report = self.prepare()
        rows = self.provenance("validation")
        self.assertEqual(len(rows), len(expected))
        self.assertEqual({row["utt_id"] for row in rows}, expected)
        for prefix in (6, 9):
            self.assertEqual({row["dataset_id"] for row in rows[:prefix]}, set(self.paths))
        for corpus in self.paths:
            early = [row for row in rows[:6] if row["dataset_id"] == corpus]
            self.assertEqual(len(early), 2)
            self.assertEqual(len({row["speaker_id"] for row in early}), 2)
        self.assertEqual(report["policy"]["validation_order"], data.VALIDATION_ORDER)
        official = data._read_rows(self.output / "validation.jsonl")
        self.assertEqual([row["audio"] for row in official], [row["audio"] for row in rows])
        train = self.provenance("train")
        keys = [(row["dataset_id"], row["utt_id"]) for row in train]
        self.assertEqual(keys, sorted(keys))
        self.assertEqual(data.audit(self.output), report)

    def test_validation_interleave_keeps_uneven_corpus_tails_exactly_once(self):
        pairs = []
        for corpus, count in (("dialogs", 2), ("scripted", 11), ("spontaneous", 5)):
            for index in range(count):
                pairs.append(({"dataset_id": corpus, "speaker_id": f"{corpus}:{index % 2}",
                               "utt_id": f"{corpus}:{index:03}"}, None))
        ordered = data._validation_order(list(reversed(pairs)))
        self.assertEqual(len(ordered), len(pairs))
        self.assertEqual({pair[0]["utt_id"] for pair in ordered}, {pair[0]["utt_id"] for pair in pairs})
        self.assertEqual({pair[0]["dataset_id"] for pair in ordered[:6]}, set(self.paths))
        self.assertEqual([pair[0]["utt_id"] for pair in ordered],
                         [pair[0]["utt_id"] for pair in data._validation_order(pairs)])

    def test_overlap_found_after_trimming_different_original_files(self):
        left = self.rows["dialogs"]["train"][0]
        right = self.rows["scripted"]["validation"][0]
        samples, rate = sf.read(left["audio"], dtype="float32")
        other = samples.copy()
        other[:1000] = .25  # Outside the selected trimmed interval.
        sf.write(right["audio"], other, rate, subtype="FLOAT")
        right["audio_sha256"] = file_digest(right["audio"])
        self.assertNotEqual(left["audio_sha256"], right["audio_sha256"])
        self.save("scripted")
        real_trim, observed = data._trim_audio, {}

        def trim(source, destination, detector):
            result = real_trim(source, destination, detector)
            if source == right["audio"]:
                # Container metadata must not defeat identical-sample detection.
                contents = bytearray(Path(destination).read_bytes())
                contents.extend(b"JUNK" + struct.pack("<I", 4) + b"test")
                struct.pack_into("<I", contents, 4, len(contents) - 8)
                Path(destination).write_bytes(contents)
                result["output_sha256"] = file_digest(destination)
            observed[source] = (result["output_sha256"], data._pcm_digest(destination))
            return result

        with patch.object(data, "_trim_audio", side_effect=trim):
            report = self.prepare()
        self.assertNotEqual(observed[left["audio"]][0], observed[right["audio"]][0])
        self.assertEqual(observed[left["audio"]][1], observed[right["audio"]][1])
        self.assertEqual(report["rejections"]["holdout_overlap"], 1)
        self.assertNotIn(left["utt_id"], {row["utt_id"] for row in self.provenance("train")})

    def test_source_session_overlap_removes_train_recordings(self):
        self.rows["dialogs"]["train"][0]["source_id"] = "shared:session"
        self.rows["scripted"]["validation"][0]["source_id"] = "shared:session"
        self.save("dialogs"); self.save("scripted")
        report = self.prepare()
        self.assertEqual(report["rejections"]["holdout_overlap"], 1)

    def test_no_speech_is_reported_and_reference_availability_rechecked(self):
        from fireredtts3_local.vad_trim import VADTrimError
        real_trim = data._trim_audio
        missing = {row["audio"] for row in self.rows["dialogs"]["train"][:3]}

        def trim(source, destination, detector):
            if source in missing:
                raise VADTrimError("no_speech", "test")
            return real_trim(source, destination, detector)

        with patch.object(data, "_trim_audio", side_effect=trim):
            with self.assertRaisesRegex(ValueError, "dialogs/train.*vad_no_speech.*no_distinct_same_speaker_reference"):
                self.prepare()
        self.assertFalse(self.output.exists())

    def test_partial_no_speech_rejections_are_reported_without_losing_corpus(self):
        from fireredtts3_local.vad_trim import VADTrimError
        real_trim = data._trim_audio
        missing = self.rows["dialogs"]["train"][0]["audio"]

        def trim(source, destination, detector):
            if source == missing:
                raise VADTrimError("no_speech", "test")
            return real_trim(source, destination, detector)

        with patch.object(data, "_trim_audio", side_effect=trim):
            report = self.prepare()
        self.assertEqual(report["rejections"], {"vad_no_speech": 1})
        self.assertEqual(report["corpora"]["dialogs"]["train"]["retained"]["clips"], 3)

    def test_scripted_long_targets_and_long_only_references_not_cropped_to_cap(self):
        for split in data.SPLITS:
            for index, row in enumerate(self.rows["scripted"][split]):
                duration = 80 + index
                samples = .12 * np.sin(np.arange(duration * 16000) * (index + 11) / (37 if split == "train" else 43))
                sf.write(row["audio"], samples, 16000, subtype="FLOAT")
                row.update(duration=duration, end=duration, audio_sha256=file_digest(row["audio"]))
        self.save("scripted")
        report = self.prepare()
        for split in data.SPLITS:
            rows = [row for row in self.provenance(split) if row["dataset_id"] == "scripted"]
            self.assertEqual(len(rows), 4)
            self.assertTrue(all(row["duration"] > 30 and row["ref_duration"] > 30 for row in rows))
            self.assertEqual(report["datasets"][split]["long_reference_pairs"], 4)

    def test_prefer_short_reference_but_keep_long_target(self):
        for split in data.SPLITS:
            row = self.rows["scripted"][split][0]
            samples = .1 * np.sin(np.arange(80 * 16000) * (1 if split == "train" else 2) / 33)
            sf.write(row["audio"], samples, 16000, subtype="FLOAT")
            row.update(duration=80, end=80, audio_sha256=file_digest(row["audio"]))
        self.save("scripted")
        self.prepare()
        for split in data.SPLITS:
            long = next(row for row in self.provenance(split) if row["utt_id"] == f"scripted:{split}:0")
            self.assertGreater(long["duration"], 30)
            self.assertLess(long["ref_duration"], 30)

    def test_original_hash_change_fails_before_output(self):
        row = self.rows["spontaneous"]["train"][0]
        Path(row["audio"]).write_bytes(Path(self.rows["spontaneous"]["train"][1]["audio"]).read_bytes())
        with self.assertRaisesRegex(ValueError, "hash changed"):
            self.prepare()
        self.assertFalse(self.output.exists())

    def test_manifest_and_derived_audio_tamper_fail_audit(self):
        self.prepare()
        row = self.provenance("train")[0]
        path = Path(row["audio"])
        samples, rate = sf.read(path)
        sf.write(path, samples * .9, rate, subtype="FLOAT")
        with self.assertRaisesRegex(ValueError, "audio changed"):
            data.audit(self.output)
        with self.assertRaisesRegex(ValueError, "audio changed"):
            self.prepare()

    def test_changed_source_manifest_and_changed_detector_not_reused(self):
        self.prepare()
        alternate = Detector()
        alternate.metadata = {"implementation": "test", "version": "2"}
        with self.assertRaisesRegex(ValueError, "different inputs/settings"):
            self.prepare(detector=alternate)
        self.rows["scripted"]["train"][0]["text"] = "Исправленная исходная расшифровка."
        self.save("scripted")
        with self.assertRaisesRegex(ValueError, "changed after preparation"):
            data.audit(self.output)

    def test_incomplete_sources_and_output_overlap_fail(self):
        with self.assertRaisesRegex(ValueError, "separate"):
            data.prepare(**self.paths, output=self.paths["dialogs"] / "child", detector=Detector())
        report_path = self.paths["scripted"] / "report.json"
        report = json.loads(report_path.read_text()); report["complete"] = False
        report_path.write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError, "complete original scripted"):
            self.prepare()

    def test_vad_source_changed_is_fatal_not_rejection(self):
        from fireredtts3_local.vad_trim import VADTrimError
        with patch.object(data, "_trim_audio", side_effect=VADTrimError("source_changed", "test")):
            with self.assertRaisesRegex(ValueError, "source_changed"):
                self.prepare()
        self.assertFalse(self.output.exists())

    def test_pinned_vad_identity_is_checked_for_prepare_and_existing_audit(self):
        from fireredtts3_local import vad_trim
        self.assertEqual(data.VAD_VERSION, vad_trim.SILERO_VAD_VERSION)
        self.assertEqual(data.VAD_MODEL_SHA256, vad_trim.SILERO_MODEL_SHA256)
        self.assertEqual(data.VAD_WHEEL_SHA256, vad_trim.SILERO_WHEEL_SHA256)
        self.assertEqual(data.VAD_SETTINGS, vad_trim.VAD_SETTINGS)
        wrong = Detector()
        wrong.metadata = {**wrong.metadata, "version": "6.1"}
        with self.assertRaisesRegex(ValueError, "pinned Silero"):
            self.prepare(detector=wrong)
        self.prepare()
        report_path = self.output / "report.json"
        report = json.loads(report_path.read_text())
        report["vad"]["model_sha256"] = "wrong"
        report_path.write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError, "pinned Silero"):
            data.audit(self.output)

    def test_trim_sample_accounting_is_checked_beyond_sidecar_hash(self):
        self.prepare()
        sidecar = self.output / "train-provenance.jsonl"
        rows = self.provenance("train")
        rows[0]["trim"]["start_sample"] += 1
        sidecar.write_text("".join(json.dumps(row) + "\n" for row in rows))
        report_path = self.output / "report.json"
        report = json.loads(report_path.read_text())
        report["manifests"][sidecar.name] = file_digest(sidecar)
        report_path.write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError, "sample accounting changed"):
            data.audit(self.output)

    def test_mp3_header_frame_estimate_does_not_change_decoded_identity_or_trim_audit(self):
        if "MP3" not in sf.available_formats():
            self.skipTest("libsndfile MP3 support is required for the compressed-source regression")
        expected_pcm = {}
        for split in data.SPLITS:
            for row in self.rows["scripted"][split]:
                samples, rate = sf.read(row["audio"], dtype="float32")
                mp3 = Path(row["audio"]).with_suffix(".mp3")
                sf.write(mp3, samples, rate, format="MP3")
                decoded, decoded_rate = sf.read(mp3, dtype="float32")
                row.update(audio=str(mp3), duration=len(decoded) / decoded_rate,
                           end=len(decoded) / decoded_rate, audio_sha256=file_digest(mp3))
                expected_pcm[str(mp3)] = data._pcm_digest(mp3)
        self.save("scripted")
        real_info = sf.info

        def approximate(path, *args, **kwargs):
            info = real_info(path, *args, **kwargs)
            if str(path).endswith(".mp3"):
                # The real Common Voice MP3s also have estimated frame counts;
                # exaggerate the difference to exceed old duration tolerances.
                return SimpleNamespace(samplerate=info.samplerate, channels=info.channels,
                                       frames=info.frames + 8000,
                                       duration=(info.frames + 8000) / info.samplerate,
                                       format=info.format, subtype=info.subtype)
            return info

        with patch.object(sf, "info", side_effect=approximate):
            report = self.prepare()
        self.assertEqual(data.audit(self.output), report)
        for split in data.SPLITS:
            for row in self.provenance(split):
                if row["dataset_id"] == "scripted":
                    self.assertEqual(row["source_pcm_sha256"], expected_pcm[row["source_audio"]])
                    samples, rate = sf.read(row["source_audio"])
                    self.assertEqual(row["trim"]["original_samples"], len(samples))
                    self.assertAlmostEqual(row["trim"]["original_duration"], len(samples) / rate)


if __name__ == "__main__":
    unittest.main()

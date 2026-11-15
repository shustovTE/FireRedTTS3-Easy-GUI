"""A duration subset keeps complete audio, holdouts, pairing and a fresh base."""
import json
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np
import soundfile as sf

from fireredtts3_local.duration_retrain import MANIFESTS, filter_duration, prepare_stage, summary
from fireredtts3_local.training_data import file_digest, load_manifest, write_manifest
from fireredtts3_local.training_features import FeatureCache
from fireredtts3_local.training_run import TrainingOptions, planned_batches, prepare_run_data
from fireredtts3_local.yo_expansion import write_json
from fireredtts3_local.yo_retrain import SCHEMA


class DurationPreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "trimmed-06"
        self.source.mkdir()
        self.output = self.root / "new-stage"
        self.groups = {}
        serial = 0
        for name, filename in MANIFESTS.items():
            rows = []
            # Exact 20s is retained. One frame over 20s must be rejected.
            durations = [20., 10., 20. + 1/24000, 19., 21.] if not name.startswith("dialogs") else [4., 6.]
            for i, duration in enumerate(durations):
                serial += 1
                audio = self.root / f"{name}-{i}.wav"
                frames = round(duration * 24000)
                sf.write(audio, np.sin(np.arange(frames) * (.01 + serial * .001)) * .1, 24000, subtype="PCM_16")
                speaker = f"yo:{name}" + ("-orphan" if i >= 3 else "")
                if name.startswith("dialogs"):
                    speaker = "dialogs:actor"
                row = dict(utt_id=f"{name}-{i}", audio=str(audio), audio_sha256=file_digest(audio),
                           speaker_id=speaker, language="Russian", source_id=name + speaker,
                           segment_source_id=f"{name}-{i}", start=0., end=frames/24000,
                           text=("Длинный текст с пробелами. " * 25 if i == 0 else f"Фраза {name} {i}."))
                if name.startswith("dialogs"):
                    row["text"] = f"Диалог {name} {i}."
                    row["dialogs_tail_review"] = dict(method="dialogs-asr-vad-tail-v1", action="keep")
                else:
                    row["tail_trim"] = dict(method=SCHEMA, padding_seconds=.2)
                rows.append(row)
            path = write_manifest(self.source / filename, rows)
            self.groups[name] = load_manifest(path)
        write_json(self.source / "preparation_state.json", dict(complete=True))
        write_json(self.source / "report.json", dict(schema=SCHEMA, complete=True, padding_seconds=.2,
            datasets={name: summary(rows) for name, rows in self.groups.items()}))
        self.config = json.loads((Path(__file__).resolve().parents[1] / "recipes/russian_trimmed_20s_base_96gb.json").read_text())
        self.config["output_dir"] = str(self.root / "training-output")
        self.recipe = self.root / "recipe.json"
        write_json(self.recipe, self.config)

    def prepare(self):
        return prepare_stage(self.source, self.output, recipe_path=self.recipe, log=lambda _: None)

    def test_actual_boundary_orphans_full_text_and_cache_are_preserved(self):
        before = {p: file_digest(p) for p in self.root.glob("*.wav")}
        report = self.prepare()
        config = json.loads((self.output / "recipe.json").read_text())
        self.assertIsNone(config["initial_adapter"])
        self.assertIsNone(config["resume_checkpoint"])
        opts = TrainingOptions.parse(config["options"])
        self.assertEqual((opts.max_prompt_seconds, opts.max_target_seconds), (20., 20.))
        with self.assertRaisesRegex(ValueError, "original unmerged"):
            opts.check_base("0" * 64)
        self.assertEqual((report["primary_examples_per_full_update"], report["replay_examples_per_full_update"]), (7, 1))
        self.assertIsNone(report["filters"]["max_text_characters"])
        for name, filename in MANIFESTS.items():
            after = load_manifest(self.output / filename)
            self.assertEqual(after, self.groups[name][:2])
            if not name.startswith("dialogs"):
                self.assertGreater(len(after[0]["text"]), 400)
                self.assertEqual(report["rejections"][name],
                    dict(duration_over_limit=2, no_distinct_reference_after_filter=1))
            cache = FeatureCache.__new__(FeatureCache)
            cache.root = Path("/cache-fixture")
            self.assertEqual(cache.path(after[0]), cache.path(self.groups[name][0]))
        self.assertEqual(before, {p: file_digest(p) for p in before})
        self.assertFalse((self.output / "audio").exists())
        train, val, replay, adaptation = prepare_run_data(config["manifest_path"], opts, config["seed"],
                                                        include_adaptation_validation=True)
        self.assertEqual((len(train.rows), len(val.rows), len(replay.rows), len(adaptation.rows)), (2, 2, 2, 2))
        batches = list(planned_batches(train, replay, opts, random.Random(1234).getstate(),
                                      0, report["total_steps"], 1, "Epochs"))
        self.assertEqual([r["utt_id"] for batch in batches for _, r, _ in batch.pairs],
                         [r["utt_id"] for r in train.rows])
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.prepare()

    def test_changed_audio_or_incomplete_source_leaves_no_output(self):
        write_json(self.source / "preparation_state.json", dict(complete=False))
        with self.assertRaisesRegex(ValueError, "preparation first"):
            self.prepare()
        self.assertFalse(self.output.exists())
        write_json(self.source / "preparation_state.json", dict(complete=True))
        row = self.groups["train"][0]
        sf.write(row["audio"], np.zeros(24000), 24000)
        with self.assertRaisesRegex(ValueError, "duration mismatch|checksum"):
            self.prepare()
        self.assertFalse(self.output.exists())

    def test_fresh_base_and_source_fingerprint_guards(self):
        self.config["initial_adapter"] = "/previous-adapter"
        write_json(self.recipe, self.config)
        with self.assertRaisesRegex(ValueError, "initial adapter"):
            self.prepare()
        self.config["initial_adapter"] = None
        write_json(self.recipe, self.config)
        rows = self.groups["train"]
        rows[0]["text"] = "Изменённая расшифровка."
        write_manifest(self.source / MANIFESTS["train"], rows)
        with self.assertRaisesRegex(ValueError, "preparation report"):
            self.prepare()
        self.assertFalse(self.output.exists())

    def test_empty_split_is_rejected_instead_of_moving_validation_speakers(self):
        with self.assertRaisesRegex(ValueError, "No usable train pairs"):
            prepare_stage(self.source, self.output, max_seconds=1., recipe_path=self.recipe, log=lambda _: None)
        self.assertFalse(self.output.exists())
        for bad in (0, -1, True, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                filter_duration([], bad)


if __name__ == "__main__":
    unittest.main()

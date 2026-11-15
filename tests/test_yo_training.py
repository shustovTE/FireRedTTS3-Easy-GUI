"""Expanded YO preparation must preserve coverage and keep Dialogs validation clean."""
import json
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np
from safetensors.torch import save_file
import soundfile as sf
import torch

from fireredtts3_local.training_data import file_digest, load_manifest, write_manifest
from fireredtts3_local.training_run import planned_batches, prepare_run_data, TrainingOptions
from fireredtts3_local.yo_expansion import write_json
from fireredtts3_local.yo_training import prepare_stage


class YoTrainingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.expansion = self.root / "expanded"
        self.expansion.mkdir()
        self.serial = 0

        def manifest(path, name, speaker, count):
            rows = []
            for i in range(count):
                self.serial += 1
                audio = self.root / f"{name}-{i}.wav"
                sf.write(audio, np.sin(np.arange(24000) * (.02 + .001 * self.serial)) * .1, 24000)
                rows.append(dict(audio=str(audio), audio_sha256=file_digest(audio), text=f"Фраза {name} {i}.",
                                 speaker_id=speaker, source_id=name, utt_id=f"{name}-{i}", language="Russian"))
            write_manifest(path, rows)
            return load_manifest(path)

        self.train = manifest(self.expansion / "train.jsonl", "yo-train", "new-train", 11)
        self.validation = manifest(self.expansion / "validation.jsonl", "yo-validation", "new-unseen", 2)
        self.replay_path = self.root / "replay.jsonl"
        self.holdout_path = self.root / "dialogs-validation.jsonl"
        self.replay = manifest(self.replay_path, "dialogs-train", "actor", 3)
        self.holdout = manifest(self.holdout_path, "dialogs-holdout", "actor", 4)
        write_json(self.expansion / "expansion_state.json", dict(complete=True, plan_sha256="fixture-plan"))
        write_json(self.expansion / "split_report.json", dict(plan_sha256="fixture-plan", train_clips=11,
                   validation_clips=2, train_hours=11/3600, validation_hours=2/3600))
        adapter = self.root / "adapter"
        adapter.mkdir()
        write_json(adapter / "adapter_config.json", dict(peft_type="LORA", r=16, lora_alpha=32,
                   lora_dropout=.05, bias="none", target_modules=["proj"]))
        write_json(adapter / "firered_lora_metadata.json", dict(base_fingerprint="fixture-base",
                   recipe=dict(options=dict(profile="backbone"))))
        save_file({"fixture": torch.zeros(1)}, adapter / "adapter_model.safetensors")
        self.recipe = json.loads((Path(__file__).resolve().parents[1] / "recipes/russian_yo_half_96gb.json").read_text())
        self.recipe["initial_adapter"] = str(adapter)
        self.recipe["output_dir"] = str(self.root / "training-output")
        self.recipe["options"]["replay_manifest"] = str(self.replay_path)
        self.recipe["options"]["adaptation_validation_manifest"] = str(self.holdout_path)
        self.recipe_path = self.root / "template.json"
        write_json(self.recipe_path, self.recipe)

    def test_filters_only_secondary_holdout_and_keeps_exact_coverage_and_resume(self):
        self.holdout[0]["text"] = self.train[0]["text"].replace(".", "?")
        self.holdout[1]["text"] = self.replay[0]["text"]
        write_manifest(self.holdout_path, self.holdout)
        inputs = [self.expansion / "train.jsonl", self.expansion / "validation.jsonl", self.holdout_path, self.replay_path]
        before = {p: p.read_bytes() for p in inputs}
        output = self.root / "stage"
        report = prepare_stage(self.expansion, output, self.recipe_path)
        self.assertEqual(before, {p: p.read_bytes() for p in inputs})
        self.assertEqual(report["datasets"]["train"]["clips"], 11)
        self.assertEqual(report["datasets"]["dialogs_validation"]["clips"], 2)
        self.assertEqual(report["dialogs_validation_excluded_text_matches"], [r["utt_id"] for r in self.holdout[:2]])
        self.assertEqual((report["updates_per_epoch"], report["total_steps"], report["replay_draws_per_epoch"]), (2, 2, 2))
        config = json.loads((output / "recipe.json").read_text())
        self.assertEqual(config["options"]["adaptation_validation_manifest"], str(output / "dialogs-validation.jsonl"))
        train, _, replay, _ = prepare_run_data(config["manifest_path"], config["options"], config["seed"],
                                              include_adaptation_validation=True)
        options = TrainingOptions.parse(config["options"])
        batches = list(planned_batches(train, replay, options, random.Random(5).getstate(), 0, 2, 1, "Epochs"))
        pairs = [p for batch in batches for p in batch.pairs]
        self.assertCountEqual([target["utt_id"] for _, target, _ in pairs if target["source_id"] == "yo-train"],
                              [r["utt_id"] for r in self.train])
        self.assertEqual(sum(target["source_id"] == "dialogs-train" for _, target, _ in pairs), 2)
        first_end = next(i for i, b in enumerate(batches) if b.sampler_state is not None)
        resumed = list(planned_batches(train, replay, options, batches[first_end].sampler_state, 1, 2, 1, "Epochs"))
        self.assertEqual(resumed, batches[first_end+1:])
        with self.assertRaisesRegex(ValueError, "new directory"):
            prepare_stage(self.expansion, output, self.recipe_path)

    def test_insufficient_clean_dialogs_pairs_leave_no_stage(self):
        for i in range(3):
            self.holdout[i]["text"] = self.train[i]["text"]
        write_manifest(self.holdout_path, self.holdout)
        output = self.root / "stage"
        with self.assertRaisesRegex(ValueError, "Too few independent Dialogs"):
            prepare_stage(self.expansion, output, self.recipe_path)
        self.assertFalse(output.exists())

    def test_incomplete_expansion_or_changed_wav_cannot_prepare_training(self):
        output = self.root / "stage"
        write_json(self.expansion / "expansion_state.json", dict(complete=False, plan_sha256="fixture-plan"))
        with self.assertRaisesRegex(ValueError, "Finish/resume"):
            prepare_stage(self.expansion, output, self.recipe_path)
        write_json(self.expansion / "expansion_state.json", dict(complete=True, plan_sha256="fixture-plan"))
        sf.write(self.train[0]["audio"], np.zeros(24000), 24000)
        with self.assertRaisesRegex(ValueError, "recorded WAV checksum"):
            prepare_stage(self.expansion, output, self.recipe_path)
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()

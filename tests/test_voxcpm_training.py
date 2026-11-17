"""No downloads/GPU: whole-recording audits and real upstream-process supervision."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import wave

from fireredtts3_local import voxcpm_training as vt


class VoxCPMTrainingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.dataset = self.root / "dataset"
        self.dataset.mkdir()
        self.model = self.root / "model"
        self.model.mkdir()
        self.source = self.root / "source"
        (self.source / "scripts").mkdir(parents=True)
        self._audio("one.wav")
        self._audio("two.wav")
        self._audio("three.wav")
        self.train = [self._row("one.wav", "two.wav"), self._row("two.wav", "one.wav")]
        self.validation = [self._row("three.wav", "one.wav")]
        self._write_dataset()
        config = {"architecture": "voxcpm2", "patch_size": 4, "feat_dim": 64, "max_length": 8192}
        for name in vt.MODEL_FILES:
            (self.model / name).write_text(json.dumps(config) if name == "config.json" else "fixture")
        vt.write_json(self.model / vt.MODEL_MARKER,
                      {"repo_id": vt.MODEL_ID, "revision": vt.MODEL_REVISION, "complete": True,
                       "files": {name: vt.digest(self.model / name) for name in vt.MODEL_FILES}})

    def _audio(self, name):
        with wave.open(str(self.dataset / name), "wb") as stream:
            stream.setnchannels(1)
            stream.setsampwidth(2)
            stream.setframerate(16000)
            stream.writeframes(b"\x00\x01" * 16000)

    def _row(self, name, ref):
        return {"audio": str(self.dataset / name), "ref_audio": str(self.dataset / ref),
                "text": "Здравствуйте, как дела?", "duration": 1.0, "ref_duration": 1.0}

    def _write_dataset(self):
        for name, rows in (("train", self.train), ("validation", self.validation)):
            (self.dataset / f"{name}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        self.report = {"complete": True,
                       "manifests": {name: vt.digest(self.dataset / name)
                                     for name in ("train.jsonl", "validation.jsonl")}}
        vt.write_json(self.dataset / "report.json", self.report)

    def _build(self, **kwargs):
        output = self.root / "recipe.yaml"
        with patch.object(vt, "verify_source", return_value=self.source), \
                patch.object(vt, "prepared_audit", return_value=self.report), \
                patch.object(vt, "tokenizer_for", return_value=lambda text: list(text)):
            result = vt.build_recipe(self.dataset, output, self.root / "output", source=self.source,
                                     model=self.model, epochs=5, grad_accum_steps=2, **kwargs)
        return output, result

    def test_recipe_uses_official_lora_and_both_splits_without_blind_1000_steps(self):
        recipe_path, result = self._build()
        recipe = vt.read_yaml(recipe_path)
        self.assertEqual(recipe["num_iters"], 5)
        self.assertEqual(recipe["max_steps"], 5)
        self.assertEqual(recipe["lora"], vt.LORA)
        self.assertEqual(recipe["sample_rate"], 16000)
        self.assertEqual(recipe["out_sample_rate"], 48000)
        self.assertEqual(Path(recipe["val_manifest"]), self.dataset / "validation.jsonl")
        self.assertTrue(result["all_targets_fit"])
        self.assertEqual(result["target_draws"], 10)
        self.assertEqual(vt.read_yaml(vt.ROOT / "recipes/voxcpm2_russian_lora.yaml")["lora"], vt.LORA)

    def test_all_validation_examples_are_length_checked_too(self):
        self.validation[0]["text"] = "а" * 200
        self._write_dataset()
        with self.assertRaisesRegex(ValueError, "Whole validation row 1.*Nothing was truncated or skipped"):
            vt.sequence_audit(self.dataset, list, context_limit=100, batch_size=1, max_batch_tokens=100)

    def test_reference_counts_toward_upstream_filter_limit(self):
        # 20 text chars + two 7-patch audio segments + 4 special tokens =38.
        with self.assertRaisesRegex(ValueError, "limit is 25"):
            vt.sequence_audit(self.dataset, lambda text: [0] * 20, context_limit=8192,
                              batch_size=2, max_batch_tokens=50)

    def test_metadata_understating_audio_duration_is_rejected(self):
        self.train[0]["duration"] = .1
        self._write_dataset()
        with self.assertRaisesRegex(ValueError, "Invalid duration metadata"):
            vt.sequence_audit(self.dataset, list, context_limit=8192, batch_size=1, max_batch_tokens=8192)

    def test_same_file_reference_and_missing_reference_are_rejected(self):
        for reference in (self.train[0]["audio"], ""):
            self.train[0]["ref_audio"] = reference
            self._write_dataset()
            with self.assertRaisesRegex(ValueError, "Distinct ref_audio"):
                vt.sequence_audit(self.dataset, list, context_limit=8192, batch_size=1, max_batch_tokens=8192)

    def test_upstream_drop_last_cannot_silently_exclude_a_target(self):
        self.train.append(self._row("three.wav", "one.wav"))
        self._write_dataset()
        with self.assertRaisesRegex(ValueError, "drop_last=True"):
            self._build(batch_size=2)

    def test_existing_recipe_and_output_are_not_overwritten(self):
        self._build()
        with self.assertRaisesRegex(ValueError, "already exists"):
            self._build()
        (self.root / "recipe.yaml").unlink()
        (self.root / "recipe.audit.json").unlink()
        output = self.root / "output"
        output.mkdir()
        (output / "latest").mkdir()
        with self.assertRaisesRegex(ValueError, "must be fresh"):
            self._build()

    def test_model_revision_and_bytes_are_both_bound(self):
        self.assertEqual(vt.verify_model(self.model)["max_length"], 8192)
        (self.model / "model.safetensors").write_bytes(b"changed weights")
        with self.assertRaisesRegex(ValueError, "changed or is missing"):
            vt.verify_model(self.model)

    @unittest.skipUnless(os.name == "posix", "Venv interpreter symlink")
    def test_venv_interpreter_symlink_is_not_resolved_out_of_environment(self):
        interpreter = self.root / "venv/bin/python"
        interpreter.parent.mkdir(parents=True)
        interpreter.symlink_to(sys.executable)
        recipe_path, _ = self._build(python=interpreter)
        audit = json.loads(recipe_path.with_suffix(".audit.json").read_text())
        self.assertEqual(audit["python"], str(interpreter))
        self.assertNotEqual(audit["python"], str(interpreter.resolve()))

    def test_recipe_edit_and_data_edit_fail_reaudit(self):
        recipe_path, _ = self._build()
        original = recipe_path.read_bytes()
        recipe_path.write_bytes(original + b"# edited\n")
        with self.assertRaisesRegex(ValueError, "Recipe changed"):
            vt.verify_recipe(recipe_path)
        recipe_path.write_bytes(original)
        (self.dataset / "train.jsonl").write_text("{}\n")
        with patch.object(vt, "verify_source", return_value=self.source):
            with self.assertRaisesRegex(ValueError, "manifests changed"):
                vt.verify_recipe(recipe_path)

    def test_changed_upstream_checkout_is_not_accepted(self):
        results = [subprocess.CompletedProcess([], 0, vt.SOURCE_COMMIT + "\n", ""),
                   subprocess.CompletedProcess([], 0, " M scripts/train_voxcpm_finetune.py\n", "")]
        with patch.object(vt, "run_checked", side_effect=results):
            with self.assertRaisesRegex(ValueError, "tracked changes"):
                vt.verify_source(self.source)

    def test_setup_does_not_replace_an_unrelated_source_directory(self):
        (self.source / "keep.txt").write_text("existing user file")
        with self.assertRaisesRegex(ValueError, "nonempty non-git"):
            vt.ensure_source(self.source)
        self.assertEqual((self.source / "keep.txt").read_text(), "existing user file")

    def test_interrupted_initial_fetch_is_retryable(self):
        source = self.root / "partial-source"
        (source / ".git").mkdir(parents=True)
        def completed(command, **kwargs):
            value = vt.SOURCE_URL if "get-url" in command else ""
            return subprocess.CompletedProcess(command, 0, value, "")
        with patch.object(vt.subprocess, "run", return_value=subprocess.CompletedProcess([], 128, "", "")), \
                patch.object(vt, "run_checked", side_effect=completed) as run, \
                patch.object(vt, "verify_source", return_value=source):
            self.assertEqual(vt.ensure_source(source), source)
        calls = [list(call.args[0]) for call in run.call_args_list]
        self.assertTrue(any("fetch" in command and vt.SOURCE_COMMIT in command for command in calls))
        self.assertTrue(any("checkout" in command for command in calls))

    @unittest.skipUnless(os.name == "posix", "POSIX job supervisor")
    def test_pilot_has_immutable_separate_output_and_original_scheduler(self):
        recipe_path, _ = self._build()
        recipe = vt.read_yaml(recipe_path)
        audit = json.loads(recipe_path.with_suffix(".audit.json").read_text())
        job = self.root / "pilot-job"
        with patch.object(vt, "ROOT", self.root), \
                patch.object(vt, "verify_recipe", return_value=(dict(recipe), dict(audit))), \
                patch.object(vt.subprocess, "Popen") as process:
            process.return_value.pid = 123
            result = vt.launch(recipe_path, job, pilot_steps=2)
        copied = vt.read_yaml(job / "recipe.yaml")
        self.assertEqual(copied["num_iters"], 2)
        self.assertEqual(copied["max_steps"], recipe["max_steps"])
        self.assertEqual(copied["warmup_steps"], recipe["warmup_steps"])
        self.assertEqual(copied["save_path"], recipe["save_path"] + "-pilot-2")
        self.assertEqual(vt.read_yaml(recipe_path), recipe)
        self.assertTrue(result["pilot"])
        self.assertTrue(process.call_args.kwargs["start_new_session"])
        self.assertTrue(process.call_args.kwargs["pass_fds"])

    @unittest.skipUnless(os.name == "posix", "POSIX job supervisor")
    def test_existing_output_refuses_upstream_implicit_resume(self):
        recipe_path, _ = self._build()
        recipe = vt.read_yaml(recipe_path)
        audit = json.loads(recipe_path.with_suffix(".audit.json").read_text())
        output = Path(recipe["save_path"])
        output.mkdir()
        (output / "latest").mkdir()
        with patch.object(vt, "ROOT", self.root), \
                patch.object(vt, "verify_recipe", return_value=(recipe, audit)), \
                patch.object(vt.subprocess, "Popen") as process:
            with self.assertRaisesRegex(ValueError, "not empty"):
                vt.launch(recipe_path, self.root / "job")
        process.assert_not_called()

    @unittest.skipUnless(os.name == "posix", "POSIX job supervisor")
    def test_worker_calls_official_script_and_reports_real_failure(self):
        recipe_path, _ = self._build()
        recipe = vt.read_yaml(recipe_path)
        audit = json.loads(recipe_path.with_suffix(".audit.json").read_text())
        marker = self.root / "called.json"
        script = self.source / "scripts/train_voxcpm_finetune.py"
        script.write_text("import json,sys\nfrom pathlib import Path\n"
                          f"Path({str(marker)!r}).write_text(json.dumps(sys.argv[1:]))\n"
                          "raise SystemExit(7)\n")
        job = self.root / "job"
        job.mkdir()
        vt.write_yaml(job / "recipe.yaml", recipe)
        vt.write_json(job / "recipe.audit.json", audit)
        vt.write_json(job / "status.json", {"cwd": str(vt.ROOT), "status": "starting"})
        fd = os.open(self.root / "lock", os.O_CREAT | os.O_RDWR, 0o600)
        with patch.object(vt, "verify_recipe", return_value=(recipe, audit)), \
                patch.object(vt, "check_training_runtime"):
            code = vt.worker(job, fd)
        self.assertEqual(code, 7)
        self.assertEqual(json.loads(marker.read_text()), ["--config_path", str(job / "recipe.yaml")])
        state = json.loads((job / "status.json").read_text())
        self.assertEqual(state["status"], "failed")
        self.assertEqual(state["exit_code"], 7)
        self.assertEqual(state["phase"], "official-train")


if __name__ == "__main__":
    unittest.main()

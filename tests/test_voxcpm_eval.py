"""Native whole-text evaluation contracts without downloading weights or a GPU."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf
import torch

from fireredtts3_local.voxcpm_eval import (_attach_adapter, adapter_info, evaluate,
                                          generate_case, load_benchmark)
from fireredtts3_local.voxcpm_training import (MODEL_FILES, MODEL_ID, MODEL_MARKER,
                                              MODEL_REVISION, digest)


class FakeNative(torch.nn.Module):
    def __init__(self, *, stop=True, error=None):
        super().__init__()
        self.stop_head = torch.nn.Linear(1, 2, bias=False)
        self.stop_head.weight.data.copy_(torch.tensor([[1.], [-1.]]))
        self.config = SimpleNamespace(max_length=8192)
        self.sample_rate = 24000
        self.patch_size = 4
        self.chunk_size = self._decode_chunk_size = 960
        self._encode_sample_rate = 24000
        self.text_tokenizer = lambda text: list(range(max(1, len(text) // 2)))
        self.calls = []
        self.stop = stop
        self.error = error

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        count = 4 if self.stop else kwargs["max_len"]
        for i in range(count):
            self.stop_head(torch.tensor([[-1. if self.stop and i == count - 1 else 1.]]))
        self.last_successful_seed = kwargs["seed"]
        return torch.linspace(-1.25, 1.25, count * 3840)[None]


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.source.mkdir()
        self.model = self.root / "model"
        self.model.mkdir()
        for name in MODEL_FILES:
            (self.model / name).write_text("{}")
        (self.model / "config.json").write_text(json.dumps({"architecture": "voxcpm2", "patch_size": 4, "feat_dim": 64}))
        (self.model / MODEL_MARKER).write_text(json.dumps({
            "repo_id": MODEL_ID, "revision": MODEL_REVISION, "complete": True,
            "files": {name: digest(self.model / name) for name in MODEL_FILES}}))
        sf.write(self.root / "reference.wav", np.sin(np.arange(16000) * .04) * .2, 16000)
        self.benchmark = self.root / "benchmark.jsonl"
        # A path-like id must not control output paths. A long multi-sentence
        # target must reach the native API without a frontend split/crop.
        self.row = {"id": "../cities", "text": "Первое предложение.\nВторое предложение? " * 35,
                    "reference_audio": "reference.wav", "reference_text": "Точный текст референса.",
                    "seed": 1235, "language": "Russian"}
        self.benchmark.write_text(json.dumps(self.row, ensure_ascii=False) + "\n")

    def run_mocked(self, tts, output=None, **kwargs):
        output = output or self.root / "evaluation"
        with patch("fireredtts3_local.voxcpm_eval.verify_source", return_value=self.source), \
             patch("fireredtts3_local.voxcpm_eval._load_model", return_value=tts):
            return evaluate(self.benchmark, self.model, output, source=self.source, **kwargs)

    def test_native_call_is_unsplit_combined_clone_single_attempt_exact_text_seed(self):
        tts = FakeNative()
        report = self.run_mocked(tts)
        self.assertTrue(report["complete"])
        self.assertTrue(report["all_generated"])
        self.assertEqual(report["execution_failures"], {})
        self.assertEqual(len(tts.calls), 1)
        call = tts.calls[0]
        self.assertEqual(call["target_text"], self.row["text"])
        self.assertGreater(len(call["target_text"]), 1000)
        self.assertEqual(call["prompt_text"], self.row["reference_text"])
        self.assertEqual(call["reference_wav_path"], call["prompt_wav_path"])
        self.assertEqual(call["reference_wav_path"], str(self.root / "reference.wav"))
        self.assertEqual(call["seed"], 1235)
        self.assertFalse(call["retry_badcase"])
        self.assertFalse(call["trim_silence_vad"])
        self.assertEqual(call["retry_badcase_max_times"], 1)
        self.assertEqual(call["retry_badcase_ratio_threshold"], 1000.)
        self.assertFalse(report["settings"]["denoise"])
        self.assertFalse(report["settings"]["split_text"])
        self.assertEqual(report["settings"]["max_generated_seconds"], 160.)
        self.assertEqual(report["results"][0]["finish_reason"], "stop")
        self.assertEqual(report["results"][0]["generated_patches"], 4)
        self.assertEqual(len(tts.stop_head._forward_hooks), 0)
        self.assertEqual(report["model_revision"], MODEL_REVISION)
        self.assertEqual(report["benchmark_sha256"], digest(self.benchmark))

    def test_raw_float_retains_out_of_range_audio_and_listening_copy_has_one_gain(self):
        report = self.run_mocked(FakeNative())
        row = report["results"][0]
        folder = self.root / "evaluation"
        raw, sr = sf.read(folder / row["float_audio"], dtype="float32")
        pcm, pcm_sr = sf.read(folder / row["audio"], dtype="float32")
        self.assertEqual((sr, pcm_sr), (24000, 24000))
        self.assertEqual(float(np.abs(raw).max()), 1.25)
        self.assertLessEqual(float(np.abs(pcm).max()), .981)
        np.testing.assert_allclose(pcm, raw * row["pcm_gain"], atol=1 / 32768)
        self.assertAlmostEqual(row["pcm_gain"], .98 / 1.25)
        self.assertEqual(row["float_audio_sha256"], digest(folder / row["float_audio"]))
        self.assertEqual(row["reference_sha256"], digest(self.root / "reference.wav"))
        self.assertGreater(row["fraction_at_or_above_full_scale"], .1)
        self.assertEqual(sf.info(folder / row["float_audio"]).subtype, "FLOAT")
        json.loads((folder / "report.json").read_text())

    def test_limit_is_reported_not_retried_or_claimed_to_be_stop(self):
        tts = FakeNative(stop=False)
        report = self.run_mocked(tts, max_steps=6)
        row = report["results"][0]
        self.assertEqual(row["finish_reason"], "max_steps")
        self.assertEqual(row["failures"], ["generation_limit"])
        self.assertEqual(len(tts.calls), 1)
        self.assertEqual(row["generated_patches"], 6)

    def test_error_remains_in_persisted_report_and_hook_removed(self):
        tts = FakeNative(error=RuntimeError("failed generation"))
        report = self.run_mocked(tts)
        self.assertTrue(report["complete"])
        self.assertFalse(report["all_generated"])
        self.assertEqual(report["results"][0]["status"], "error")
        self.assertIn("failed generation", report["results"][0]["error"])
        self.assertEqual(len(tts.calls), 1)
        self.assertEqual(len(tts.stop_head._forward_hooks), 0)
        self.assertEqual(list((self.root / "evaluation").glob("*.wav")), [])

    def test_context_overflow_is_reported_without_truncation_or_native_call(self):
        tts = FakeNative()
        tts.config.max_length = 20
        report = self.run_mocked(tts)
        self.assertIn("nothing was truncated", report["results"][0]["error"])
        self.assertEqual(tts.calls, [])

    def test_output_is_immutable_and_seed_override_is_explicit(self):
        tts = FakeNative()
        report = self.run_mocked(tts, seed=777)
        self.assertEqual(tts.calls[0]["seed"], 777)
        self.assertEqual(report["results"][0]["seed"], 777)
        before = (self.root / "evaluation/report.json").read_bytes()
        with self.assertRaises(FileExistsError):
            self.run_mocked(FakeNative())
        self.assertEqual((self.root / "evaluation/report.json").read_bytes(), before)

    def test_invalid_reference_duplicate_id_and_seed_rejected_before_generation(self):
        for row in (dict(self.row, seed=True), dict(self.row, seed=-1), dict(self.row, reference_text=" ")):
            self.benchmark.write_text(json.dumps(row) + "\n")
            with self.assertRaises(ValueError):
                load_benchmark(self.benchmark)
        self.benchmark.write_text((json.dumps(self.row) + "\n") * 2)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            load_benchmark(self.benchmark)
        self.benchmark.write_text(json.dumps(self.row) + "\n")
        sf.write(self.root / "reference.wav", np.zeros((1000, 2)), 16000)
        with self.assertRaisesRegex(ValueError, "mono"):
            load_benchmark(self.benchmark)

    def test_changed_model_receipt_hash_prevents_loading(self):
        (self.model / "model.safetensors").write_bytes(b"changed weights")
        with self.assertRaisesRegex(ValueError, "model file changed"):
            self.run_mocked(FakeNative())
        self.assertFalse((self.root / "evaluation").exists())

    def test_official_adapter_identity_and_strict_parameter_loading(self):
        adapter = self.root / "adapter"
        adapter.mkdir()
        (adapter / "lora_weights.safetensors").write_bytes(b"weights")
        config = {"base_model": str(self.model), "lora_config": {"r": 32, "enable_lm": True}}
        (adapter / "lora_config.json").write_text(json.dumps(config))
        info = adapter_info(adapter, self.model)
        class AdapterModel:
            def named_parameters(self):
                return [("layer.lora_A", None), ("layer.lora_B", None), ("base.weight", None)]
            def load_lora_weights(self, filename):
                return ["layer.lora_A", "layer.lora_B"], []
        _attach_adapter(AdapterModel(), info)
        partial = AdapterModel()
        partial.load_lora_weights = lambda _: (["layer.lora_A"], [])
        with self.assertRaisesRegex(ValueError, "partial adapter"):
            _attach_adapter(partial, info)
        config["base_model"] = "unrelated/model"
        (adapter / "lora_config.json").write_text(json.dumps(config))
        with self.assertRaisesRegex(ValueError, "different base"):
            adapter_info(adapter, self.model)


if __name__ == "__main__":
    unittest.main()

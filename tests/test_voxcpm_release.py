from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import torch
from safetensors.torch import load_file, save_file

from fireredtts3_local import voxcpm_release as release


class VoxCPMReleaseTest(unittest.TestCase):
    def test_separate_native_branches_match_saved_linear_and_missing_dit_is_rejected(self):
        cfg = dict(release.LORA, target_modules_lm=["q_proj"], target_modules_dit=["o_proj"])
        generator = torch.Generator().manual_seed(44)
        roots = ["base_lm.attn.q_proj", "residual_lm.attn.q_proj", "feat_decoder.estimator.attn.o_proj"]
        def rand(*shape):
            return torch.randn(shape, generator=generator)
        x = rand(7, 11)
        weights = {"untrained.weight": rand(8, 4)}
        adapter = {}
        expected = {}
        for root in roots:
            w, a, b = rand(5, 11), rand(32, 11), rand(5, 32)
            weights[root + ".weight"] = w
            adapter[root + ".lora_A"], adapter[root + ".lora_B"] = a, b
            expected[root] = (x @ w.T) + ((x @ a.T) @ b.T)
        with tempfile.TemporaryDirectory() as tmp:
            base, lora, output = (Path(tmp) / n for n in ("base", "lora", "out"))
            save_file(weights, base)
            save_file(adapter, lora)
            self.assertEqual(release.merge_weights(base, lora, cfg, output)["changed_layers"], 3)
            merged = load_file(output)
            self.assertEqual(set(merged), set(weights))
            self.assertTrue(torch.equal(merged["untrained.weight"], weights["untrained.weight"]))
            for root, target in expected.items():
                torch.testing.assert_close(x @ merged[root + ".weight"].T, target, atol=3e-5, rtol=1e-5)
            del adapter[roots[-1] + ".lora_A"]
            save_file(adapter, lora)
            with self.assertRaisesRegex(ValueError, "partial merge"):
                release.merge_weights(base, lora, cfg, Path(tmp) / "bad")

    def test_public_or_unrelated_destination_never_receives_files(self):
        for private, files in [(False, []), (True, [Mock(rfilename="unrelated.txt")])]:
            with self.subTest(private=private):
                api = Mock()
                api.model_info.return_value = Mock(private=private, siblings=files)
                with self.assertRaises(ValueError):
                    release.upload_private(Path("."), api)
                api.upload_folder.assert_not_called()

    def test_full_export_can_be_reused_after_upload_interruption_but_rejects_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base, checkpoint, output = root / "base", root / "latest", root / "export"
            base.mkdir()
            checkpoint.mkdir()
            for asset in release.MODEL_FILES:
                (base / asset).write_text(asset)
            release.write_json(base / release.MODEL_MARKER, {"files": {
                asset: release.digest(base / asset) for asset in release.MODEL_FILES}})
            (root / "LICENSE").write_text("upstream license")
            code = root / release.TOKENIZER_CODE
            code.write_text("tokenizer code")
            weights = checkpoint / "lora_weights.safetensors"
            weights.write_text("training weights")
            release.write_json(checkpoint / "training_state.json", {"step": release.STEP})
            adapter = {"weights": str(weights), "config": release.LORA, "files": {weights.name: release.digest(weights)}}
            def merge(src, weights, config, dest):
                dest.write_bytes(src.read_bytes() + b"-merged")
                return {"target_layers": 192, "changed_layers": 192}
            with (patch.object(release, "DEFAULT_MODEL", base), patch.object(release, "CHECKPOINT", checkpoint),
                  patch.object(release, "verify_model"), patch.object(release, "verify_source", return_value=root),
                  patch.object(release, "adapter_info", return_value=adapter),
                  patch.object(release, "hf_hub_download", return_value=code),
                  patch.object(release, "TOKENIZER_SHA", release.digest(code)),
                  patch.object(release, "merge_weights", side_effect=merge) as merger):
                first = release.prepare(output)
                self.assertEqual(release.prepare(output), first)
                merger.assert_called_once()
                self.assertEqual(set(first["files"]), release.PAYLOAD)
                (output / "audiovae.pth").write_text("changed")
                with self.assertRaisesRegex(ValueError, "Existing export differs"):
                    release.prepare(output)

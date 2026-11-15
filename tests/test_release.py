"""Real PEFT merge, CKermit packaging, and publication without network writes."""
import copy
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import httpx
from huggingface_hub.errors import RepositoryNotFoundError
from peft import LoraConfig, get_peft_model
from safetensors.torch import load_file, save_file
import torch

from fireredtts3_local.export import TOKENIZER_FILES, export_merged
from fireredtts3_local.model_manager import ModelManager
from fireredtts3_local.release import (MANIFEST, TEMPLATE_REPO, TEMPLATE_REVISION, WEIGHTS,
    build_release, git_blob_digest, prepare_template, publish_release, seal_release, verify_release, write_json)
from fireredtts3_local.training_data import file_digest, model_identity
from fireredtts3_local.training_objective import adapter_targets


def file_info(path, name):
    path = Path(path)
    return SimpleNamespace(rfilename=name, size=path.stat().st_size,
        lfs=SimpleNamespace(sha256=file_digest(path)) if path.suffix in {".safetensors", ".bin"} else None,
        blob_id=git_blob_digest(path))


class ReleaseTests(unittest.TestCase):
    def setUp(self):
        ModelManager._ensure_upstream_namespace()
        import fireredtts3.llm.fireredtts3_base as upstream
        from fireredtts3.redae.redae import RedAEConfig
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.base, self.adapter, self.template = [self.root / n for n in ("base", "adapter", "template")]
        self.template.mkdir()
        cfg = dict(upstream.Qwen3_1_7B_ConfigDict)
        cfg.update(hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=4,
                   num_key_value_heads=2, head_dim=8, vocab_size=64, attn_implementation="sdpa")
        context = patch.object(upstream, "Qwen3_1_7B_ConfigDict", cfg)
        context.start(); self.addCleanup(context.stop)
        torch.manual_seed(42)
        core = upstream.FireRedTTS3BaseCore(upstream.FireRedTTS3BaseCoreConfig(
            redae_dim=4, patch_size=2, spk_in_dim=6, patch_encoder_hidden_size=16,
            patch_encoder_depth=1, patch_encoder_num_heads=2, dit_hidden_size=16, dit_depth=1, dit_num_heads=2))
        core.save_pretrained(self.base / "fireredtts3_base")
        redae = RedAEConfig(bottleneck_dim=4)
        redae.save_pretrained(self.base / "redae")
        save_file({"encoder.fixture": torch.ones(2)}, self.base / "redae/model.safetensors")
        (self.base / "campp").mkdir()
        (self.base / "campp/campplus_voxceleb.bin").write_bytes(b"fixture-speaker-weights")
        (self.base / "text_tokenizer").mkdir()
        write_json(self.base / "text_tokenizer/tokenizer_config.json", {})
        self.config = {**redae.to_dict(), **core.config.to_dict(), **core.backbone_llm.config.to_dict(),
            "model_type": "fireredtts3", "architectures": ["FireRedTTS3ForConditionalGeneration"],
            "rope_theta": cfg["rope_theta"], "dit_in_channels": 26, "redae_scale": 1., "use_lm_head": False}
        write_json(self.template / "config.json", self.config)
        (self.template / "README.md").write_text("Upstream Apache-2.0 model card")
        for name in TOKENIZER_FILES:
            (self.template / name).write_text("{}" if name.endswith(".json") else "fixture")
        self.original = {k: v.clone() for k, v in core.state_dict().items()}
        adapted = get_peft_model(core, LoraConfig(r=2, lora_alpha=4, lora_dropout=.05,
                                                target_modules=adapter_targets(core, "backbone_dit")))
        for name, parameter in adapted.named_parameters():
            if "lora_B" in name:
                parameter.data.fill_(.02)
        adapted.save_pretrained(self.adapter)
        write_json(self.adapter / "firered_lora_metadata.json", {"base_fingerprint": model_identity(self.base),
                   "recipe": {"options": {"profile": "backbone_dit"}}})
        self.inventory = [file_info(self.template / n, n) for n in ["config.json", "README.md", *TOKENIZER_FILES]]
        self.inventory += [file_info(self.base / relative, n) for n, relative in WEIGHTS.items()]

    def download(self, cache, names, repo, revision, **kwargs):
        self.assertEqual((repo, revision, kwargs["repo_type"], kwargs["authenticated"]),
                         (TEMPLATE_REPO, TEMPLATE_REVISION, "model", True))
        self.assertFalse(set(names) & set(WEIGHTS))
        cache = Path(cache); cache.mkdir(parents=True, exist_ok=True)
        for name in names:
            shutil.copy2(self.template / name, cache / name)

    def build(self, output="release", **kwargs):
        with patch("huggingface_hub.HfApi") as api, \
             patch("fireredtts3_local.release.download_files", side_effect=self.download), \
             patch("fireredtts3_local.release.verify_tokenizer", return_value={"backend_equal": True}):
            api.return_value.model_info.return_value = SimpleNamespace(sha=TEMPLATE_REVISION, siblings=self.inventory)
            return build_release(self.base, self.adapter, self.root / output,
                                 template_cache=self.root / "cache", log=lambda _: None, **kwargs)

    def remote(self, root, private=False):
        return SimpleNamespace(sha="b" * 40, private=private,
            siblings=[file_info(p, p.relative_to(root).as_posix()) for p in root.rglob("*") if p.is_file()])

    def test_package_keeps_ckermit_assets_and_checks_every_tensor(self):
        result = self.build()
        root = self.root / "release"
        manifest = verify_release(root)
        self.assertEqual(result["release_id"], manifest["release_id"])
        weights = load_file(root / "model.safetensors")
        self.assertEqual(weights.keys(), self.original.keys())
        self.assertFalse(torch.equal(weights["dit_head.weight"], self.original["dit_head.weight"]))
        torch.testing.assert_close(weights["spk_proj_dit.weight"], self.original["spk_proj_dit.weight"], rtol=0, atol=0)
        self.assertEqual((root / "tokenizer_config.json").read_bytes(), (self.template / "tokenizer_config.json").read_bytes())
        self.assertEqual((root / "redae/model.safetensors").read_bytes(), (self.base / "redae/model.safetensors").read_bytes())
        provenance = json.loads((root / "export_provenance.json").read_text())
        self.assertTrue(provenance["saved_tensor_hashes_verified"])
        self.assertGreater(provenance["layer_probes"], 7)
        self.assertGreater(provenance["changed_tensors"], 7)
        self.assertFalse(provenance["sglang_runtime_validated"])
        self.assertNotIn(str(self.root), json.dumps(provenance))
        self.assertFalse(any("trainer_state" in p or ".cache" in p for p in manifest["files"]))
        self.assertEqual(file_digest(self.base / "fireredtts3_base/model.safetensors"),
                         next(f.lfs.sha256 for f in self.inventory if f.rfilename == "model.safetensors"))

    def test_wrong_template_base_fails_before_download(self):
        next(f for f in self.inventory if f.rfilename == "model.safetensors").lfs.sha256 = "0" * 64
        with patch("huggingface_hub.HfApi") as api, patch("fireredtts3_local.release.download_files") as download:
            api.return_value.model_info.return_value = SimpleNamespace(sha=TEMPLATE_REVISION, siblings=self.inventory)
            with self.assertRaisesRegex(ValueError, "Template/base weights differ"):
                prepare_template(self.base, TEMPLATE_REPO, TEMPLATE_REVISION, self.root / "cache", lambda _: None)
            download.assert_not_called()

    def test_backbone_only_adapter_keeps_all_acoustic_and_speaker_tensors(self):
        from fireredtts3.llm.fireredtts3_base import FireRedTTS3BaseCore
        core = FireRedTTS3BaseCore.from_pretrained(self.base / "fireredtts3_base")
        model = get_peft_model(core, LoraConfig(r=2, lora_alpha=4, lora_dropout=.05,
                                              target_modules=adapter_targets(core, "backbone")))
        for name, parameter in model.named_parameters():
            if "lora_B" in name:
                parameter.data.fill_(.01)
        model.save_pretrained(self.adapter)
        write_json(self.adapter / "firered_lora_metadata.json", {"base_fingerprint": model_identity(self.base),
                   "recipe": {"options": {"profile": "backbone"}}})
        self.build()
        weights = load_file(self.root / "release/model.safetensors")
        changed = {k for k in weights if not torch.equal(weights[k], self.original[k])}
        self.assertEqual(len(changed), 7)
        self.assertTrue(all(k.startswith("backbone_llm.") for k in changed))
        for key in self.original:
            if not key.startswith("backbone_llm."):
                torch.testing.assert_close(weights[key], self.original[key], rtol=0, atol=0)

    def test_failed_merge_is_atomic_and_keeps_original_unchanged(self):
        weights = load_file(self.adapter / "adapter_model.safetensors")
        key = next(k for k in weights if "lora_B" in k)
        weights[key].flatten()[0] = float("nan")
        save_file(weights, self.adapter / "adapter_model.safetensors")
        before = model_identity(self.base)
        with self.assertRaises(ValueError):
            export_merged(self.base, self.adapter, self.template / "config.json", self.root / "bad", log=lambda _: None)
        self.assertFalse((self.root / "bad").exists())
        self.assertEqual(model_identity(self.base), before)

    def test_evaluation_summary_does_not_publish_reference_text_or_paths(self):
        report = {"benchmark_sha256": "test", "speaker_model": "test-scorer", "asr_model": "test-asr", "complete": True,
                  "results": [{"id": "private-case", "speaker_id": "private-speaker", "speaker_similarity": .8,
                               "wer": .0, "text": "private transcript", "reference_audio": "/private/voice.wav"}]}
        write_json(self.root / "baseline.json", report)
        report = copy.deepcopy(report)
        report["adapter"] = str(self.adapter)
        report["results"][0]["speaker_similarity"] = .81
        write_json(self.root / "adapted.json", report)
        self.build(evaluation_baseline=self.root / "baseline.json", evaluation_adapted=self.root / "adapted.json")
        root = self.root / "release"
        text = (root / "README.md").read_text() + (root / "export_provenance.json").read_text()
        self.assertNotIn("private transcript", text)
        self.assertNotIn("/private/voice.wav", text)
        self.assertNotIn("private-speaker", text)
        self.assertIn("0.01000000", text)

    def test_publish_uses_token_owner_explicit_allowlist_and_parent_commit(self):
        self.build()
        root = self.root / "release"
        empty = SimpleNamespace(sha="a" * 40, private=False, siblings=[])
        remote = self.remote(root)
        with patch("huggingface_hub.HfApi") as client:
            api = client.return_value
            api.whoami.return_value = {"name": "fixture-owner"}
            response = httpx.Response(404, request=httpx.Request("GET", "https://huggingface.co/api/models/fixture-owner/model"))
            api.model_info.side_effect = [RepositoryNotFoundError("missing", response=response), empty, remote]
            api.upload_folder.return_value = SimpleNamespace(oid=remote.sha)
            result = publish_release(root, public=True, log=lambda _: None)
            client.assert_called_once_with(token=True)
            api.create_repo.assert_called_once_with("fixture-owner/FireRedTTS3-Base-Russian", repo_type="model",
                                                   private=False, exist_ok=False)
            args = api.upload_folder.call_args.kwargs
            self.assertEqual(args["parent_commit"], "a" * 40)
            self.assertEqual(set(args["allow_patterns"]), {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()})
            self.assertNotIn("delete_patterns", args)
            self.assertEqual(result["commit"], remote.sha)
            self.assertTrue(result["remote_files_verified"])
        self.assertTrue((root.parent / "release.published.json").is_file())
        verify_release(root)  # The upload receipt is outside the sealed package.
        with patch("huggingface_hub.HfApi") as client:
            api = client.return_value
            api.whoami.return_value = {"name": "fixture-owner"}
            api.model_info.return_value = remote
            publish_release(root, public=True, log=lambda _: None)
            api.create_repo.assert_not_called()
            api.upload_folder.assert_not_called()

    def test_changed_local_or_existing_remote_files_are_not_uploaded(self):
        self.build()
        root = self.root / "release"
        remote = self.remote(root)
        next(f for f in remote.siblings if f.rfilename == "model.safetensors").lfs.sha256 = "0" * 64
        with patch("huggingface_hub.HfApi") as client:
            api = client.return_value
            api.whoami.return_value = {"name": "fixture-owner"}
            api.model_info.return_value = remote
            with self.assertRaisesRegex(ValueError, "different files"):
                publish_release(root, public=True, log=lambda _: None)
            api.upload_folder.assert_not_called()
        (root / ".env").write_text("fixture-secret")
        with patch("huggingface_hub.HfApi") as client, self.assertRaisesRegex(ValueError, "missing/extra"):
            publish_release(root, public=True, log=lambda _: None)
        client.assert_not_called()
        (root / ".env").unlink()
        (root / "tokenizer.json").write_text("changed")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            verify_release(root)


class PublishUpdateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.old, self.new = self.root / "old", self.root / "new"
        for name in ("README.md", "LICENSE", "config.json", "tokenizer.json", "export_provenance.json",
                     "redae/model.safetensors", "campp/campplus_voxceleb.bin",
                     "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors",
                     "model.safetensors.index.json"):
            path = self.old / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}" if name.endswith(".json") else f"old fixture {name}")
        seal_release(self.old)
        shutil.copytree(self.old, self.new)
        for path in self.new.glob("model*"):
            path.unlink()
        (self.new / "model.safetensors").write_bytes(b"new merged core fixture")
        (self.new / "README.md").write_text("New release card")
        seal_release(self.new)
        self.attributes = self.root / "attributes"
        self.attributes.write_text("*.safetensors filter=lfs diff=lfs merge=lfs -text\n")
        self.old_info = self.remote(self.old, "a" * 40)
        self.new_info = self.remote(self.new, "b" * 40)
        self.context = patch("huggingface_hub.HfApi")
        self.api = self.context.start().return_value
        self.addCleanup(self.context.stop)
        self.api.whoami.return_value = {"name": "shustov-te"}
        self.api.model_info.side_effect = [self.old_info, self.new_info]
        self.api.hf_hub_download.side_effect = lambda **kw: str(self.old / kw["filename"])
        self.api.create_commit.return_value = SimpleNamespace(oid=self.new_info.sha)

    def remote(self, root, sha):
        siblings = [file_info(p, p.relative_to(root).as_posix()) for p in root.rglob("*") if p.is_file()]
        siblings.append(file_info(self.attributes, ".gitattributes"))
        return SimpleNamespace(sha=sha, private=False, siblings=siblings)

    def publish(self, **kwargs):
        return publish_release(self.new, repo_id="shustov-te/FireRedTTS3-Base-Russian", public=True,
                               update_existing=True, log=lambda _: None, **kwargs)

    def test_atomic_commit_removes_obsolete_shards_and_index_preserves_assets_and_retry(self):
        from huggingface_hub import CommitOperationAdd, CommitOperationDelete
        result = self.publish(expected_parent=self.old_info.sha)
        args = self.api.create_commit.call_args.kwargs
        self.assertEqual(args["repo_id"], "shustov-te/FireRedTTS3-Base-Russian")
        self.assertEqual(args["parent_commit"], self.old_info.sha)
        self.assertEqual(args["revision"], "main")
        removed = {op.path_in_repo for op in args["operations"] if isinstance(op, CommitOperationDelete)}
        self.assertEqual(removed, {"model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors",
                                   "model.safetensors.index.json"})
        added = {op.path_in_repo for op in args["operations"] if isinstance(op, CommitOperationAdd)}
        self.assertEqual(added, {"model.safetensors", "README.md", MANIFEST})
        self.assertEqual(result["previous_commit"], self.old_info.sha)
        self.assertEqual(result["commit"], self.new_info.sha)
        self.api.upload_folder.assert_not_called()
        self.api.create_repo.assert_not_called()
        downloads = [call.kwargs for call in self.api.hf_hub_download.call_args_list]
        self.assertTrue(all(d["revision"] == self.old_info.sha for d in downloads))
        self.assertFalse(any(d["filename"].endswith((".safetensors", ".bin")) for d in downloads))
        self.api.create_commit.reset_mock()
        self.api.model_info.side_effect = None
        self.api.model_info.return_value = self.new_info
        # Same release after a successful/lost response is an idempotent retry,
        # even if expected_parent still refers to the old HEAD.
        self.assertEqual(self.publish(expected_parent=self.old_info.sha), result)
        self.api.create_commit.assert_not_called()
        verify_release(self.new)

    def test_nonmanaged_or_edited_repositories_are_never_overwritten(self):
        for case in ("missing_manifest", "unmanaged_file", "changed_weight", "changed_small_file"):
            with self.subTest(case=case):
                info = copy.deepcopy(self.old_info)
                if case == "missing_manifest":
                    info.siblings = [f for f in info.siblings if f.rfilename != MANIFEST]
                elif case == "unmanaged_file":
                    info.siblings.append(file_info(self.attributes, "custom-notes.txt"))
                elif case == "changed_weight":
                    next(f for f in info.siblings if f.rfilename.startswith("model-")).lfs.sha256 = "0" * 64
                else:
                    next(f for f in info.siblings if f.rfilename == "config.json").blob_id = "0" * 40
                self.api.model_info.side_effect = [info]
                with self.assertRaises(ValueError):
                    self.publish()
                self.api.create_commit.assert_not_called()
                self.assertFalse((self.root / "new.published.json").exists())

    def test_wrong_owner_parent_and_concurrent_push_fail_without_receipt(self):
        self.api.whoami.return_value = {"name": "another-owner"}
        with self.assertRaisesRegex(ValueError, "local HF token"):
            self.publish()
        self.api.model_info.assert_not_called()
        self.api.whoami.return_value = {"name": "shustov-te"}
        with self.assertRaisesRegex(ValueError, "expected_parent"):
            self.publish(expected_parent="c" * 40)
        self.api.create_commit.assert_not_called()
        self.api.model_info.side_effect = [self.old_info]
        self.api.create_commit.side_effect = RuntimeError("409 concurrent parent change")
        with self.assertRaisesRegex(RuntimeError, "concurrent"):
            self.publish()
        self.assertFalse((self.root / "new.published.json").exists())

    def test_remote_verification_failure_does_not_write_success_receipt(self):
        broken = copy.deepcopy(self.new_info)
        next(f for f in broken.siblings if f.rfilename == "model.safetensors").lfs.sha256 = "0" * 64
        self.api.model_info.side_effect = [self.old_info, broken]
        with self.assertRaisesRegex(ValueError, "remote content verification failed"):
            self.publish()
        self.assertFalse((self.root / "new.published.json").exists())


if __name__ == "__main__":
    unittest.main()

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fireredtts3_local import workspace_paths as paths


class WorkspacePathsTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.old, self.new = self.root / "old", self.root / "new"
        self.new.mkdir()
        for name, value in (("WORKSPACE_ROOT", self.new), ("LEGACY_ROOT", self.old)):
            patched = patch.object(paths, name, value)
            patched.start()
            self.addCleanup(patched.stop)

    def test_maps_only_exact_prefix_without_resolving_python_symlink(self):
        (self.new / "bin").mkdir()
        (self.new / "bin/python").symlink_to("/usr/bin/python3")
        self.assertEqual(paths.local_path(self.old / "bin/python"), self.new / "bin/python")
        self.assertEqual(paths.local_path(self.root / "old-other/a.wav"), self.root / "old-other/a.wav")
        self.assertEqual(paths.local_path("relative.wav"), Path("relative.wav"))
        with self.assertRaisesRegex(ValueError, "escape"):
            paths.local_path(self.old / "../outside")
        self.assertFalse(self.old.exists())

    def test_recorded_identity_does_not_require_old_directory(self):
        self.assertEqual(paths.recorded_path(self.new / "a.wav", self.old / "data"), self.old / "a.wav")
        self.assertEqual(paths.recorded_path(self.new / "a.wav", self.new / "data"), self.new / "a.wav")
        self.assertEqual(paths.recorded_path(self.root / "external.wav", self.old / "data"), self.root / "external.wav")

    def test_runtime_view_keeps_original_bytes_text_and_order(self):
        for name in ("a.wav", "b.wav"):
            (self.new / name).write_bytes(name.encode())
        rows = [dict(audio=str(self.old / "a.wav"), ref_audio="b.wav", text="Куда?\nСюда.", duration=1),
                dict(audio=str(self.old / "b.wav"), ref_audio=str(self.old / "a.wav"), text="Да!", duration=2)]
        manifest = self.new / "train.jsonl"
        original = "".join(json.dumps(row) + "\n" for row in rows).encode()
        manifest.write_bytes(original)
        view = paths.runtime_manifest(self.old / "train.jsonl")
        actual = [json.loads(line) for line in view.read_text().splitlines()]
        self.assertEqual([r["text"] for r in actual], [r["text"] for r in rows])
        self.assertEqual([r["duration"] for r in actual], [1, 2])
        self.assertEqual(actual[0]["audio"], str(self.new / "a.wav"))
        self.assertEqual(actual[0]["ref_audio"], str(self.new / "b.wav"))
        self.assertEqual(manifest.read_bytes(), original)
        self.assertEqual(paths.runtime_manifest(manifest), view)
        self.assertFalse(self.old.exists())
        view.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "Runtime manifest changed"):
            paths.runtime_manifest(manifest)

    def test_current_canonical_manifest_does_not_need_copy(self):
        for name in ("a.wav", "b.wav"):
            (self.new / name).touch()
        manifest = self.new / "train.jsonl"
        manifest.write_text(json.dumps(dict(audio=str(self.new / "a.wav"), ref_audio=str(self.new / "b.wav"))))
        self.assertEqual(paths.runtime_manifest(manifest), manifest)
        self.assertFalse((self.new / ".runtime").exists())


if __name__ == "__main__":
    unittest.main()

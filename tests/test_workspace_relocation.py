import importlib.util
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
import venv


spec = importlib.util.spec_from_file_location(
    "relocate_workspace", Path(__file__).resolve().parents[1] / "scripts/relocate_workspace.py")
relocation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(relocation)


class RelocationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.new = self.root / "new"
        (self.new / "fireredtts3_local").mkdir(parents=True)
        (self.new / "README.md").write_text("fixture")
        self.old = self.root / "old"

    def test_old_paths_work_and_contents_are_unchanged(self):
        manifest = self.new / "manifest.jsonl"
        original = ('{"audio": "' + str(self.old / "voice.wav") + '"}\n').encode()
        manifest.write_bytes(original)
        (self.new / "voice.wav").write_bytes(b"immutable fixture")
        self.assertEqual(relocation.restore_alias(self.old, self.new)["status"], "linked")
        self.assertEqual((self.old / "voice.wav").read_bytes(), b"immutable fixture")
        self.assertEqual((self.old / "manifest.jsonl").read_bytes(), original)
        self.assertEqual(relocation.restore_alias(self.old, self.new)["status"], "already_linked")

    def test_existing_directory_and_foreign_or_dangling_links_are_not_replaced(self):
        self.old.mkdir()
        sentinel = self.old / "keep"
        sentinel.write_text("untouched")
        with self.assertRaisesRegex(ValueError, "Refusing to replace"):
            relocation.restore_alias(self.old, self.new)
        self.assertEqual(sentinel.read_text(), "untouched")
        sentinel.unlink()
        self.old.rmdir()
        for target in (self.root, self.root / "missing"):
            self.old.symlink_to(target, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "Refusing to replace"):
                relocation.restore_alias(self.old, self.new)
            self.assertEqual(self.old.readlink(), target)
            self.old.unlink()

    def test_same_directory_is_noop_and_in_checkout_alias_is_rejected(self):
        self.assertEqual(relocation.restore_alias(self.new, self.new)["status"], "already_at_destination")
        with self.assertRaisesRegex(ValueError, "outside the current checkout"):
            relocation.restore_alias(self.new / "alias", self.new)
        self.assertFalse((self.new / "alias").exists())

    def test_wrong_destination_and_missing_parent_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "not a FireRed"):
            relocation.restore_alias(self.old, self.root)
        with self.assertRaisesRegex(ValueError, "parent directory does not exist"):
            relocation.restore_alias(self.root / "missing" / "old", self.new)
        self.assertFalse((self.root / "missing").exists())

    @unittest.skipUnless(os.name == "posix", "POSIX interpreter/shebang relocation")
    def test_moved_venv_editable_path_and_old_shebang_still_work(self):
        environment = self.new / ".venv"
        venv.EnvBuilder(with_pip=False, symlinks=True).create(environment)
        python = environment / "bin/python"
        purelib = Path(subprocess.check_output(
            [str(python), "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"], text=True).strip())
        source = self.new / "source"
        source.mkdir()
        (source / "relocation_fixture.py").write_text("VALUE = 'editable import OK'\n")
        (purelib / "relocation-fixture.pth").write_text(str(source) + "\n")
        command = environment / "bin/fixture-command"
        command.write_text(f"#!{python}\nimport relocation_fixture; print(relocation_fixture.VALUE)\n")
        command.chmod(0o755)
        moved = self.root / "moved"
        self.new.rename(moved)
        relocation.restore_alias(self.new, moved)
        for argv in ([str(moved / ".venv/bin/python"), "-c",
                      "import relocation_fixture; print(relocation_fixture.VALUE)"],
                     [str(moved / ".venv/bin/fixture-command")]):
            self.assertEqual(subprocess.check_output(argv, text=True).strip(), "editable import OK")


if __name__ == "__main__":
    unittest.main()

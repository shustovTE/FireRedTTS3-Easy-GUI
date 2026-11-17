"""Integration regression for hosts without TorchCodec-compatible FFmpeg.

Run with .venv-voxcpm2/bin/python. Other environments skip this optional upstream
integration rather than downloading models/dependencies during unit tests.
"""
import importlib.util
import subprocess
import sys
import unittest

from fireredtts3_local.voxcpm_training import ROOT


@unittest.skipUnless(importlib.util.find_spec("voxcpm"), "Run in the official VoxCPM environment")
class AudioRuntimeTests(unittest.TestCase):
    def test_training_and_native_audio_work_when_torchcodec_import_fails(self):
        script = '''
import builtins
import importlib
real_import = builtins.__import__
real_import_module = importlib.import_module
def reject_codec(name, *args, **kwargs):
    if name == "torchcodec" or name.startswith("torchcodec."):
        raise RuntimeError("Simulated TorchCodec failure: missing libavutil.so.59")
    return real_import(name, *args, **kwargs)
def reject_codec_module(name, *args, **kwargs):
    if name == "torchcodec" or name.startswith("torchcodec."):
        raise RuntimeError("Simulated TorchCodec failure: missing libavutil.so.59")
    return real_import_module(name, *args, **kwargs)
builtins.__import__ = reject_codec
importlib.import_module = reject_codec_module
from fireredtts3_local.voxcpm_training import check_audio_runtime
result = check_audio_runtime()
assert result["audio_runtime"] == "ok"
assert result["sample_rate"] == 16000
assert result["torchcodec_required_for_this_workflow"] is False
print("Audio runtime passed with broken TorchCodec")
'''
        result = subprocess.run([sys.executable, "-c", script], cwd=ROOT, text=True,
                                capture_output=True, timeout=90)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("Audio runtime passed with broken TorchCodec", result.stdout)


if __name__ == "__main__":
    unittest.main()

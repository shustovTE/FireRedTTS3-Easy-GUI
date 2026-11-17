"""Pinned offline DPDFNet8 preprocessing for post-VAD Spontaneous WAVs."""
from importlib.metadata import version
from pathlib import Path

import numpy as np
import soundfile as sf

from .training_data import file_digest

RATE = 16000
MODEL_REVISION = "dd6818d00f50c836fed43a6243ebe49116de5964"
POLICY = {"package": "dpdfnet", "version": "0.6.0", "model": "dpdfnet8",
          "repo": "Ceva-IP/DPDFNet", "revision": MODEL_REVISION, "file": "onnx/dpdfnet8.onnx",
          "sha256": "899d4f23f3ff86edbffa8c537e4bcbdc49da1b4e84e0ef390611e0604a3b26cb",
          "sample_rate": RATE, "attn_limit_db": 15.0, "provider": "CPUExecutionProvider",
          "right_context_samples": 640, "format": "WAV/FLOAT"}


def frames_at_16k(frames, rate):
    return (frames * RATE + rate - 1) // rate


class SpontaneousDenoiser:
    def __init__(self):
        from dpdfnet.onnx_backend import build_runtime_model
        from huggingface_hub import hf_hub_download

        if version("dpdfnet") != POLICY["version"]:
            raise ValueError("Install dpdfnet==0.6.0 in the VoxCPM environment")
        path = Path(hf_hub_download(POLICY["repo"], POLICY["file"], revision=MODEL_REVISION))
        if file_digest(path) != POLICY["sha256"]:
            raise ValueError("DPDFNet8 ONNX weights differ from the pinned SHA256")
        self.runtime = build_runtime_model(path)

    def __call__(self, source, destination):
        # The pinned package's offline kernel reuses the CPU session and starts
        # each utterance with a fresh copy of the model's recurrent state.
        from dpdfnet.api import _enhance_with_runtime
        from dpdfnet.audio import ensure_sample_rate

        cut, rate = sf.read(source, dtype="float32")
        if cut.ndim != 1 or not cut.size or not np.isfinite(cut).all():
            raise ValueError(f"Invalid post-VAD mono waveform: {source}")
        samples = ensure_sample_rate(cut, rate, RATE)
        if len(samples) != frames_at_16k(len(cut), rate):
            raise ValueError("DPDFNet8 input resampling changed the expected length")
        # Flush the offline ISTFT lookahead outside the cut so that its trailing
        # zero fill cannot replace the last consonant of a tightly cut recording.
        context = POLICY["right_context_samples"]
        padded = np.pad(samples, (0, context))
        result = _enhance_with_runtime(padded, RATE, runtime=self.runtime,
                                       model_sample_rate=RATE, attn_limit_db=POLICY["attn_limit_db"])
        if result.shape != padded.shape or not np.isfinite(result).all() or not result[:len(samples)].any():
            raise ValueError(f"Invalid DPDFNet8 output: {source}")
        sf.write(destination, result[:len(samples)], RATE, subtype="FLOAT", format="WAV")

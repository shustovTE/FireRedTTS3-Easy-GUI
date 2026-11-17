"""Trim only the outside silence of a recording with pinned, offline Silero VAD.

Detection uses a resampled copy. The output is an untouched contiguous slice of
the original-rate samples: no ASR, gain adjustment, internal cuts or duration cap.
"""
from __future__ import annotations

import copy
import hashlib
import io
import math
import os
import tempfile
import threading
from importlib import metadata, resources
from numbers import Integral
from pathlib import Path
from typing import Protocol

import numpy as np
import soundfile as sf
import torch
import torchaudio


SILERO_VAD_VERSION = "6.2.3"
# PyPI silero_vad-6.2.3-py3-none-any.whl, verified before implementation.
SILERO_WHEEL_SHA256 = "7b7f5436cfcb02fae583a05b512ea96467fd449fe54cb49a5e4f06c51a1e43b8"
SILERO_MODEL_SHA256 = "e1122837f4154c511485fe0b9c64455f7b929c96fbb8d79fbdb336383ebd3720"
VAD_SAMPLE_RATE = 16000
VAD_SETTINGS = {
    "threshold": 0.5,
    "neg_threshold": 0.35,
    # Do not throw away short detected words at the outside of the recording.
    "min_speech_duration_ms": 0,
    "min_silence_duration_ms": 100,
    "speech_pad_ms": 0,
    "max_speech_duration_s": None,  # JSON-safe representation of no limit.
    "sampling_rate": VAD_SAMPLE_RATE,
    "return_seconds": False,
}


class VADTrimError(ValueError):
    """An explicit rejected recording, distinct from installation / I/O errors."""

    def __init__(self, reason: str, message: str):
        self.reason = reason
        super().__init__(f"{reason}: {message}")


class Detector(Protocol):
    metadata: dict

    def __call__(self, audio_16k: torch.Tensor) -> list[dict]: ...


class SileroDetector:
    """Reusable CPU detector; uses only the verified model bundled in 6.2.3.

    Install ``silero-vad==6.2.3`` beforehand. No torch.hub or network requests are
    made here. Calls are serialized because the recurrent model has state.
    """

    def __init__(self):
        try:
            installed = metadata.version("silero-vad")
        except metadata.PackageNotFoundError:
            raise RuntimeError("Install silero-vad==6.2.3 before VAD preparation") from None
        if installed != SILERO_VAD_VERSION:
            raise RuntimeError(f"Expected silero-vad=={SILERO_VAD_VERSION}; found {installed}")
        # silero_vad.model changes PyTorch's global thread count on import. Keep
        # the caller's thread policy instead of silently changing other work.
        threads = torch.get_num_threads()
        try:
            from silero_vad import get_speech_timestamps
        finally:
            if torch.get_num_threads() != threads:
                torch.set_num_threads(threads)
        blob = resources.files("silero_vad.data").joinpath("silero_vad.jit").read_bytes()
        digest = hashlib.sha256(blob).hexdigest()
        if digest != SILERO_MODEL_SHA256:
            raise RuntimeError("Bundled Silero 6.2.3 JIT model SHA256 mismatch; reinstall the pinned package")
        # Load the bytes whose digest was checked, not an unchecked second path.
        self._model = torch.jit.load(io.BytesIO(blob), map_location="cpu").eval()
        self._get_timestamps = get_speech_timestamps
        self._lock = threading.Lock()
        self.metadata = {
            "package": "silero-vad", "version": installed, "backend": "torchscript",
            "device": "cpu", "model_resource": "silero_vad.data/silero_vad.jit",
            "model_sha256": digest, "pypi_wheel_sha256": SILERO_WHEEL_SHA256,
            "torch_version": str(torch.__version__), "settings": copy.deepcopy(VAD_SETTINGS),
        }

    def __call__(self, audio_16k: torch.Tensor) -> list[dict]:
        if (not torch.is_tensor(audio_16k) or audio_16k.ndim != 1
                or audio_16k.device.type != "cpu" or not audio_16k.numel()):
            raise VADTrimError("invalid_audio", "detector requires nonempty mono CPU audio at 16000 Hz")
        if not torch.isfinite(audio_16k).all():
            raise VADTrimError("nonfinite_audio", "detector input contains NaN or infinity")
        settings = dict(VAD_SETTINGS, max_speech_duration_s=float("inf"))
        with self._lock, torch.inference_mode():
            return self._get_timestamps(audio_16k.float().contiguous(), self._model, **settings)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _intervals(detected, samples: int) -> list[dict]:
    if not isinstance(detected, (list, tuple)):
        raise VADTrimError("invalid_boundaries", "VAD must return a list of sample intervals")
    if not detected:
        raise VADTrimError("no_speech", "VAD found no speech; no output was created")
    result, previous_end = [], 0
    for segment in detected:
        if not isinstance(segment, dict) or "start" not in segment or "end" not in segment:
            raise VADTrimError("invalid_boundaries", "VAD interval needs start and end samples")
        start, end = segment["start"], segment["end"]
        if (isinstance(start, bool) or isinstance(end, bool)
                or not isinstance(start, Integral) or not isinstance(end, Integral)
                or not previous_end <= start < end <= samples):
            raise VADTrimError("invalid_boundaries", "VAD intervals must be integral, ordered, disjoint and inside the audio")
        result.append({"start": int(start), "end": int(end)})
        previous_end = end
    return result


def trim_audio(source, destination, detector: Detector | None = None,
               padding_seconds: float = .2) -> dict:
    """Write a new FLOAT WAV and return sample-exact provenance.

    ``start_sample`` and exclusive ``end_sample`` address the original recording.
    At most ``padding_seconds`` outside nonspeech is retained on either edge,
    according to VAD boundaries. Interior pauses/samples are never removed.
    A supplied detector accepts a 1-D CPU tensor at 16 kHz and returns integer
    sample intervals on that grid; its ``metadata`` identifies it in the report.
    Existing destinations, including symlinks, are never overwritten.
    """
    if (isinstance(padding_seconds, bool) or not isinstance(padding_seconds, (int, float))
            or not math.isfinite(padding_seconds) or padding_seconds < 0):
        raise ValueError("padding_seconds must be finite and nonnegative")
    source = Path(source).resolve(strict=True)
    destination = Path(os.path.abspath(destination))
    if source == destination.resolve():
        raise ValueError("VAD output must differ from the immutable source")
    if destination.suffix.lower() != ".wav":
        raise ValueError("VAD output must be a .wav file")
    if os.path.lexists(destination):
        raise FileExistsError(f"VAD output already exists: {destination}")
    original_hash = _sha256(source)
    try:
        info = sf.info(source)
        audio, sample_rate = sf.read(source, dtype="float32", always_2d=True)
    except (sf.LibsndfileError, ValueError) as exc:
        raise VADTrimError("invalid_audio", f"Cannot decode source audio: {source}") from exc
    if audio.shape[1] != 1 or not len(audio) or sample_rate <= 0:
        raise VADTrimError("invalid_audio", "Expected nonempty mono audio with a positive sample rate")
    audio = audio[:, 0]
    if not np.isfinite(audio).all():
        raise VADTrimError("nonfinite_audio", "Source contains NaN or infinity")
    # libsndfile estimates MP3 frame counts from the compressed stream header;
    # valid VBR / encoder-padding files can decode to a different sample count.
    # Decode is authoritative for MP3. PCM / lossless containers stay strict.
    if sample_rate != info.samplerate or (info.format != "MP3" and len(audio) != info.frames):
        raise VADTrimError("invalid_audio", "Decoded samples differ from source audio metadata")
    vad_audio = torch.from_numpy(audio.copy())
    if sample_rate != VAD_SAMPLE_RATE:
        vad_audio = torchaudio.functional.resample(vad_audio, sample_rate, VAD_SAMPLE_RATE)
    if not torch.isfinite(vad_audio).all():
        raise VADTrimError("nonfinite_audio", "VAD resampling produced nonfinite audio")
    detector = SileroDetector() if detector is None else detector
    if not isinstance(getattr(detector, "metadata", None), dict):
        raise ValueError("VAD detector must expose a metadata dict identifying its version/settings")
    vad_intervals = _intervals(detector(vad_audio), len(vad_audio))
    # Conservative integer conversion: do not cut a detected fractional source
    # sample. The final resampled grid point can extend past the source by <1
    # VAD sample because resampling returns ceil(original_samples * rate_ratio).
    speech = [{"start": min(len(audio), s["start"] * sample_rate // VAD_SAMPLE_RATE),
               "end": min(len(audio), (s["end"] * sample_rate + VAD_SAMPLE_RATE - 1) // VAD_SAMPLE_RATE)}
              for s in vad_intervals]
    if any(s["start"] >= s["end"] for s in speech):
        raise VADTrimError("invalid_boundaries", "Speech interval is empty on the original sample grid")
    padding = math.floor(padding_seconds * sample_rate)
    start = max(0, speech[0]["start"] - padding)
    end = min(len(audio), speech[-1]["end"] + padding)
    if not 0 <= start < end <= len(audio):
        raise VADTrimError("invalid_boundaries", "Trim boundaries fall outside source samples")
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    os.close(fd)
    temporary = Path(temporary)
    try:
        sf.write(temporary, audio[start:end], sample_rate, format="WAV", subtype="FLOAT")
        output_hash = _sha256(temporary)
        if _sha256(source) != original_hash:
            raise VADTrimError("source_changed", "Source changed while processing; no output was published")
        result = {
            "schema": "silero-outside-trim-v1", "source": str(source), "destination": str(destination),
            "source_sha256": original_hash, "output_sha256": output_hash,
            "sample_rate": sample_rate, "original_samples": len(audio), "output_samples": end - start,
            "start_sample": start, "end_sample": end,
            "original_duration": len(audio) / sample_rate, "duration": (end - start) / sample_rate,
            "leading_removed_samples": start, "trailing_removed_samples": len(audio) - end,
            "trimmed": start != 0 or end != len(audio), "padding_seconds": float(padding_seconds),
            "padding_samples": padding, "source_format": info.format, "source_subtype": info.subtype,
            "source_info_frames": info.frames, "source_info_frames_difference": len(audio) - info.frames,
            "source_frame_count_policy": "decoded_authoritative_mp3" if info.format == "MP3" else "exact_match",
            "output_format": "WAV", "output_subtype": "FLOAT",
            "speech_intervals": speech, "vad_speech_intervals": vad_intervals,
            "vad_input_samples": len(vad_audio), "vad": copy.deepcopy(detector.metadata),
            "resampling": {"for_detection_only": True, "source_rate": sample_rate,
                           "vad_rate": VAD_SAMPLE_RATE, "implementation": "torchaudio.functional.resample",
                           "torchaudio_version": str(torchaudio.__version__),
                           "boundary_rounding": "floor_start_ceil_end_clamped_to_source"},
        }
        # Same-directory hard link publishes the complete file atomically and
        # fails if another process has created this destination in the meantime.
        os.link(temporary, destination)
        return result
    finally:
        temporary.unlink(missing_ok=True)

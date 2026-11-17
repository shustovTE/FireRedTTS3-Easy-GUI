"""Outside-only VAD trims preserve samples, original files and provenance."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf
import torch

from fireredtts3_local.vad_trim import (
    SILERO_MODEL_SHA256, SILERO_VAD_VERSION, SileroDetector, VADTrimError, trim_audio,
)


class FixedDetector:
    def __init__(self, intervals):
        self.intervals = intervals
        self.calls = []
        self.metadata = {"package": "test-detector", "version": "1", "settings": {"sampling_rate": 16000}}

    def __call__(self, audio):
        self.calls.append(audio.clone())
        return self.intervals


class TrimTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def recording(self, samples, rate=16000, filename="source.wav"):
        path = self.root / filename
        sf.write(path, np.asarray(samples, dtype=np.float32), rate, subtype="FLOAT")
        return path

    def test_trim_keeps_both_200ms_margins_and_internal_pause_exactly(self):
        # Detection is mocked so this verifies sample slicing independently of
        # VAD accuracy. The interior pause must not be assembled away.
        rate = 16000
        original = np.linspace(-1.2, 1.2, rate * 5, dtype=np.float32)
        original[rate * 2:rate * 3] = 0
        source = self.recording(original)
        source_bytes = source.read_bytes()
        detector = FixedDetector([{"start": rate, "end": rate * 2}, {"start": rate * 3, "end": rate * 4}])
        output = self.root / "out" / "trimmed.wav"
        report = trim_audio(source, output, detector)
        actual, sr = sf.read(output, dtype="float32")
        self.assertEqual((report["start_sample"], report["end_sample"]), (12800, 67200))
        np.testing.assert_array_equal(actual, original[12800:67200])
        self.assertEqual(sr, rate)
        self.assertEqual(source.read_bytes(), source_bytes)
        self.assertEqual(report["source_sha256"], hashlib.sha256(source_bytes).hexdigest())
        self.assertEqual(report["output_sha256"], hashlib.sha256(output.read_bytes()).hexdigest())
        self.assertEqual(report["output_samples"], len(actual))
        self.assertEqual(report["duration"], len(actual) / rate)
        self.assertEqual(sf.info(output).subtype, "FLOAT")
        self.assertEqual(report["vad"], detector.metadata)
        json.dumps(report, allow_nan=False)

    def test_exactly_or_less_than_200ms_nonspeech_is_not_trimmed(self):
        original = np.linspace(-.8, .8, 32000, dtype=np.float32)
        for leading, trailing in ((3200, 3200), (1000, 2000), (0, 0)):
            with self.subTest(leading=leading, trailing=trailing):
                source = self.recording(original, filename=f"source-{leading}.wav")
                output = self.root / f"trimmed-{leading}.wav"
                result = trim_audio(source, output, FixedDetector([{"start": leading, "end": len(original)-trailing}]))
                self.assertFalse(result["trimmed"])
                self.assertEqual((result["start_sample"], result["end_sample"]), (0, len(original)))
                np.testing.assert_array_equal(sf.read(output, dtype="float32")[0], original)

    def test_single_sample_over_padding_is_trimmed_without_gain_change(self):
        original = np.full(32000, 1.25, np.float32)
        source = self.recording(original)
        output = self.root / "trimmed.wav"
        result = trim_audio(source, output, FixedDetector([{"start": 3201, "end": 28799}]))
        self.assertEqual((result["leading_removed_samples"], result["trailing_removed_samples"]), (1, 1))
        np.testing.assert_array_equal(sf.read(output, dtype="float32")[0], original[1:-1])

    def test_resampling_is_detection_only_at_24_and_48khz(self):
        for rate in (24000, 48000):
            with self.subTest(rate=rate):
                original = (np.sin(np.arange(rate * 3) * .03) * .5).astype(np.float32)
                source = self.recording(original, rate, f"source-{rate}.wav")
                detector = FixedDetector([{"start": 8001, "end": 32001}])
                output = self.root / f"trimmed-{rate}.wav"
                result = trim_audio(source, output, detector)
                first = 8001 * rate // 16000
                last = (32001 * rate + 15999) // 16000
                start, end = first - rate // 5, last + rate // 5
                self.assertEqual(detector.calls[0].shape, (48000,))
                self.assertEqual(detector.calls[0].device.type, "cpu")
                self.assertEqual((result["start_sample"], result["end_sample"]), (start, end))
                actual, actual_rate = sf.read(output, dtype="float32")
                self.assertEqual(actual_rate, rate)
                np.testing.assert_array_equal(actual, original[start:end])

    def test_resample_ceil_at_eof_does_not_invent_source_samples(self):
        source = self.recording(np.ones(24001, np.float32) * .25, 24000)
        detector = FixedDetector([{"start": 0, "end": 16001}])
        report = trim_audio(source, self.root / "out.wav", detector)
        self.assertEqual(report["output_samples"], 24001)
        self.assertEqual(report["speech_intervals"], [{"start": 0, "end": 24001}])

    def test_mp3_header_frame_estimate_uses_decoded_samples_and_records_difference(self):
        original = np.linspace(-.5, .5, 16000, dtype=np.float32)
        source = self.recording(original)
        # Simulate the metadata of a valid MP3: real Scripted files return an
        # estimated header frame count different from sf.read's decoded length.
        header = SimpleNamespace(frames=16097, samplerate=16000, format="MP3", subtype="MPEG_LAYER_III")
        with patch("fireredtts3_local.vad_trim.sf.info", return_value=header):
            result = trim_audio(source, self.root / "mp3-decoded.wav",
                                FixedDetector([{"start": 0, "end": 16000}]))
        self.assertEqual(result["original_samples"], 16000)
        self.assertEqual(result["source_info_frames"], 16097)
        self.assertEqual(result["source_info_frames_difference"], -97)
        self.assertEqual(result["source_frame_count_policy"], "decoded_authoritative_mp3")
        np.testing.assert_array_equal(sf.read(result["destination"], dtype="float32")[0], original)

    def test_lossless_frame_metadata_mismatch_remains_an_error(self):
        source = self.recording(np.zeros(16000, np.float32))
        for fmt in ("WAV", "FLAC"):
            with self.subTest(format=fmt):
                header = SimpleNamespace(frames=16097, samplerate=16000, format=fmt, subtype="PCM_16")
                with patch("fireredtts3_local.vad_trim.sf.info", return_value=header):
                    with self.assertRaises(VADTrimError) as raised:
                        trim_audio(source, self.root / f"{fmt}.wav", FixedDetector([]))
                self.assertEqual(raised.exception.reason, "invalid_audio")
                self.assertFalse((self.root / f"{fmt}.wav").exists())

    def test_long_clip_has_no_30_or_60_second_cap(self):
        source = self.recording(np.zeros(16000 * 75, np.float32))
        result = trim_audio(source, self.root / "out.wav",
                            FixedDetector([{"start": 16000, "end": 16000 * 74}]))
        self.assertEqual(result["duration"], 73.4)
        self.assertEqual(result["original_duration"], 75.)

    def test_no_speech_nonfinite_and_stereo_are_explicit_rejections(self):
        cases = [(np.zeros(16000), [], "no_speech"),
                 (np.full(16000, np.nan), [{"start": 0, "end": 16000}], "nonfinite_audio"),
                 (np.full(16000, np.inf), [], "nonfinite_audio"),
                 (np.zeros((16000, 2)), [], "invalid_audio"),
                 (np.zeros(0), [], "invalid_audio")]
        for i, (audio, intervals, reason) in enumerate(cases):
            with self.subTest(reason=reason, i=i):
                source = self.recording(audio, filename=f"bad-{i}.wav")
                output = self.root / f"out-{i}.wav"
                with self.assertRaises(VADTrimError) as raised:
                    trim_audio(source, output, FixedDetector(intervals))
                self.assertEqual(raised.exception.reason, reason)
                self.assertFalse(output.exists())

    def test_invalid_vad_intervals_never_publish_output(self):
        source = self.recording(np.zeros(16000))
        bad = [None, [{"start": -1, "end": 10}], [{"start": 0, "end": 16001}],
               [{"start": 5, "end": 5}], [{"start": 5.0, "end": 10}],
               [{"start": False, "end": 10}], [{"start": 5}],
               [{"start": 5, "end": 15}, {"start": 10, "end": 20}],
               [{"start": 50, "end": 60}, {"start": 10, "end": 20}]]
        for i, intervals in enumerate(bad):
            with self.subTest(intervals=intervals):
                output = self.root / f"out-{i}.wav"
                with self.assertRaises(VADTrimError) as raised:
                    trim_audio(source, output, FixedDetector(intervals))
                self.assertEqual(raised.exception.reason, "invalid_boundaries")
                self.assertFalse(output.exists())

    def test_source_and_existing_outputs_cannot_be_overwritten(self):
        source = self.recording(np.zeros(16000))
        original = source.read_bytes()
        detector = FixedDetector([{"start": 0, "end": 16000}])
        with self.assertRaises(ValueError):
            trim_audio(source, source, detector)
        output = self.root / "out.wav"
        output.write_bytes(b"existing artifact")
        with self.assertRaises(FileExistsError):
            trim_audio(source, output, detector)
        self.assertEqual(output.read_bytes(), b"existing artifact")
        self.assertEqual(source.read_bytes(), original)
        broken = self.root / "broken.wav"
        broken.symlink_to(self.root / "absent.wav")
        with self.assertRaises(FileExistsError):
            trim_audio(source, broken, detector)

    def test_concurrently_changed_source_is_detected_before_publish(self):
        source = self.recording(np.zeros(16000))
        output = self.root / "out.wav"
        class MutatingDetector(FixedDetector):
            def __call__(inner, audio):
                with source.open("ab") as stream:
                    stream.write(b"source-changed")
                return super().__call__(audio)
        with self.assertRaises(VADTrimError) as raised:
            trim_audio(source, output, MutatingDetector([{"start": 0, "end": 16000}]))
        self.assertEqual(raised.exception.reason, "source_changed")
        self.assertFalse(output.exists())
        self.assertFalse(list(self.root.glob(".out.wav.*.tmp")))

    def test_invalid_padding_is_not_silently_clamped(self):
        source = self.recording(np.zeros(16000))
        for value in (-.1, float("nan"), float("inf"), True, "0.2"):
            with self.subTest(padding=value), self.assertRaises(ValueError):
                trim_audio(source, self.root / "out.wav", FixedDetector([]), value)

    def test_rejects_wrong_installed_version_before_importing_model(self):
        with patch("fireredtts3_local.vad_trim.metadata.version", return_value="6.2.2"):
            with self.assertRaisesRegex(RuntimeError, "Expected silero-vad==6.2.3"):
                SileroDetector()

    def test_rejects_missing_package_without_network_fallback(self):
        with patch("fireredtts3_local.vad_trim.metadata.version", side_effect=importlib.metadata.PackageNotFoundError):
            with self.assertRaisesRegex(RuntimeError, "Install silero-vad==6.2.3"):
                SileroDetector()


def pinned_model_available():
    try:
        return importlib.metadata.version("silero-vad") == SILERO_VAD_VERSION
    except importlib.metadata.PackageNotFoundError:
        return False


@unittest.skipUnless(pinned_model_available(), "optional real-model smoke requires silero-vad==6.2.3")
class PinnedModelTests(unittest.TestCase):
    def test_verified_cpu_model_reuses_state_safely_for_silence(self):
        previous_threads = torch.get_num_threads()
        try:
            torch.set_num_threads(1)
            detector = SileroDetector()
            self.assertEqual(detector.metadata["model_sha256"], SILERO_MODEL_SHA256)
            self.assertEqual(detector.metadata["settings"]["speech_pad_ms"], 0)
            self.assertIsNone(detector.metadata["settings"]["max_speech_duration_s"])
            self.assertEqual(torch.get_num_threads(), 1)
            for _ in range(2):
                self.assertEqual(detector(torch.zeros(16000)), [])
            with patch("fireredtts3_local.vad_trim.SILERO_MODEL_SHA256", "0" * 64):
                with self.assertRaisesRegex(RuntimeError, "SHA256 mismatch"):
                    SileroDetector()
        finally:
            torch.set_num_threads(previous_threads)


if __name__ == "__main__":
    unittest.main()

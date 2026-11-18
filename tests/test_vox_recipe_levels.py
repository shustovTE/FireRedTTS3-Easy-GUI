import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

from fireredtts3_local import vox_recipe_levels as levels, vox_recipe_transfer as transfer, voxcpm_data
from fireredtts3_local.training_data import file_digest
from fireredtts3_local.vox_recipe_compare import default_benchmark


class LevelTests(unittest.TestCase):
    def test_both_drifts_decrease_and_pauses_stay_quiet(self):
        sr = 8000
        positions = np.arange(sr * 18) / sr
        ramp = 10 ** (np.linspace(0, 10, len(positions)) / 20)
        x = .1 * np.sin(2*np.pi*200*positions) * ramp
        x[7*sr:9*sr] = 0
        for samples in (x, x[::-1]):
            y, stats, drift = levels.level_audio(samples, sr)
            self.assertEqual(y.shape, x.shape)
            self.assertTrue(np.all(y[samples == 0] == 0))
            self.assertTrue(np.all(np.abs(y) <= np.abs(samples) + 1e-7))
            self.assertLess(abs(drift(y)), abs(stats['active_drift_before_db']) - 2)
            self.assertLessEqual(stats['max_envelope_cut_db'], 6)

    def test_steady_level_and_silence(self):
        x = np.tile(np.array([.25, -.25], dtype='float32'), 16000)
        for samples in (x, x*0, x*1e-6, x[:1]):
            y, stats, _ = levels.level_audio(samples, 16000)
            np.testing.assert_array_equal(y, samples)
            self.assertEqual(stats['max_envelope_cut_db'], 0)

    @unittest.skipUnless(shutil.which('ffmpeg'), 'FFmpeg is required for the R128 measurement')
    def test_common_loudness_on_decoded_wav_including_high_peaks(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name, amplitude, spike in [('soft', .006, False), ('loud', .65, False), ('transient', .03, True),
                                           ('brief', .1, False)]:
                sr = 16000 if name == 'soft' else 44100
                t = np.arange(sr//5 if name == 'brief' else 4*sr)/sr
                samples = amplitude*np.sin(2*np.pi*220*t)
                if spike:
                    samples[sr] = .98
                path, dest = root/f'{name}.wav', root/f'{name}-out.wav'
                sf.write(path, samples, sr, subtype='PCM_24')
                stats = levels.normalize_recording(path, dest)
                y, output_rate = sf.read(dest)
                self.assertEqual((len(y), output_rate), (len(samples), sr))
                lu, tp = levels.measure_loudness(y, sr)
                self.assertAlmostEqual(lu, -23., delta=.2)
                self.assertLessEqual(tp, -1.)
                self.assertEqual(stats['peak_limiter_used'], spike)
                if spike:
                    self.assertEqual(int(np.argmax(np.abs(y))), sr)
            sf.write(path, np.zeros(16000), 16000)
            with self.assertRaisesRegex(ValueError, 'Silent'):
                levels.normalize_recording(path, dest)

    @unittest.skipUnless(shutil.which('ffmpeg'), 'FFmpeg is required for the R128 measurement')
    def test_all_wavs_but_identical_pairs_and_generation_references(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, output = root / 'source', root / 'levelled'
            source.mkdir()
            (source / 'report.json').write_text('{}')
            for split, lengths in [('train', [9, 7, 5, 3]), ('validation', [8, 5])]:
                rows = []
                for i, sec in enumerate(lengths):
                    name = f'{split}-{i}'
                    path = source / f'{name}.wav'
                    t = np.arange(1000*sec)/1000
                    samples = .08 * np.exp(t/8) * np.sin(2*np.pi*(80+i)*t)
                    sf.write(path, samples, 1000)
                    sha = file_digest(path)
                    pcm = voxcpm_data._pcm_digest(path)
                    rows.append(dict(utt_id=name, dataset_id='dialogs', speaker_id='dialogs:one', duration=sec,
                                     text=f'{name}?', audio=str(path), audio_sha256=sha, pcm_sha256=pcm,
                                     source_audio=str(path), source_sha256=sha, source_pcm_sha256=pcm))
                for i, row in enumerate(rows):
                    ref = rows[(i+1) % len(rows)]
                    row.update(ref_audio=ref['audio'], ref_text=ref['text'], ref_sha256=ref['audio_sha256'],
                               ref_pcm_sha256=ref['pcm_sha256'], ref_duration=ref['duration'], ref_utt_id=ref['utt_id'])
                (source / f'{split}-provenance.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            with patch.object(voxcpm_data, 'audit', return_value={'schema': voxcpm_data.FAIR_SCHEMA}), \
                    patch.object(transfer, 'CAP_SECONDS', 20):
                original, _, old = transfer.paired_data(source)
                path = levels.prepare(source, output)
                self.assertEqual(path, levels.prepare(source, output))
                new, unique, selected = transfer.paired_data(output)
                self.assertEqual(len(json.loads(path.read_text())['files']), 6)
                self.assertEqual(len(unique), 4)
                self.assertEqual(selected['level_summary']['train']['clips'], 2)
                self.assertEqual(selected['original_selection_sha256'], old['selection_sha256'])
                self.assertNotEqual(selected['selection_sha256'], old['selection_sha256'])
                for split, pairs in new.items():
                    self.assertEqual([(p['utt_id'], t['utt_id']) for p,t,_ in pairs],
                                     [(p['utt_id'], t['utt_id']) for p,t,_ in original[split]])
                    for p, target, _ in pairs[:8]:
                        self.assertEqual(p['audio'], target['ref_audio'])
                        self.assertEqual(p['audio_sha256'], target['ref_sha256'])
                        self.assertEqual(p['pcm_sha256'], target['ref_pcm_sha256'])
                for row in unique:
                    self.assertEqual(row['audio_sha256'], file_digest(row['audio']))
                    self.assertEqual(row['unlevelled_audio_sha256'], file_digest(row['unlevelled_audio']))
                    self.assertEqual(sf.info(row['audio']).frames, sf.info(row['unlevelled_audio']).frames)
                self.assertEqual(default_benchmark(new['validation']), default_benchmark(original['validation']))
                Path(unique[0]['audio']).write_bytes(b'invalid WAV')
                with self.assertRaisesRegex(ValueError, 'WAV changed'):
                    transfer.paired_data(output)

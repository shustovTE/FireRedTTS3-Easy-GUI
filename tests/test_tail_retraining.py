"""End-of-word crops, acoustic review, split preservation and fresh adapters."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf
import torch
import torch.nn.functional as F

from fireredtts3_local.dialogs_tail_review import decide_tail, review_dialogs
from fireredtts3_local.training_batch import stop_diagnostics
from fireredtts3_local.training_data import file_digest, load_manifest, write_manifest
from fireredtts3_local.training_features import FeatureCache
from fireredtts3_local.training_run import TrainingOptions
from fireredtts3_local.yo_dataset import aligned_segments, import_yo
from fireredtts3_local.yo_expansion import REPO, REVISION, write_json
from fireredtts3_local.yo_retrain import prepare_retraining, crop_row


class TailBoundaryTests(unittest.TestCase):
    def record(self):
        return dict(text='привет мир', text_denorm='Привет, мир!', duration=4.,
            text_alignment='привет(0.1,0.5) [pause0.4](0.5,0.9) мир(0.9,1.2) [pause2.8](1.2,4.0)')

    def test_final_pause_ignored_internal_pause_and_full_text_kept(self):
        row = self.record()
        self.assertEqual(aligned_segments(row, segmentation='complete-utterance'), [(0., 4., 'Привет, мир!')])
        self.assertEqual(aligned_segments(row, segmentation='complete-utterance', tail_padding_seconds=.2),
                         [(0., 1.4, 'Привет, мир!')])
        short = dict(row, duration=1.3, text_alignment='привет(0.1,0.5) мир(0.9,1.2)')
        self.assertEqual(aligned_segments(short, segmentation='complete-utterance', tail_padding_seconds=.2)[0][1], 1.3)
        for bad in (-.1, float('nan'), float('inf'), True):
            with self.assertRaises(ValueError):
                aligned_segments(row, segmentation='complete-utterance', tail_padding_seconds=bad)
        with self.assertRaises(ValueError):
            aligned_segments(row, tail_padding_seconds=.2)
        with self.assertRaisesRegex(ValueError, 'alignment_outside_audio'):
            aligned_segments(dict(row, duration=.8), segmentation='complete-utterance', tail_padding_seconds=.2)

    def test_direct_import_keeps_trim_after_actual_duration_correction(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            records = []
            for i in range(2):
                sf.write(root/f'{i}.wav', np.sin(np.arange(96000)*(.01+i*.001))*.1, 24000, subtype='PCM_16')
                records.append(dict(self.record(), audio=f'{i}.wav', duration=3.98,
                    utt_id=str(i), local_spk_id='a', lang='ru', mos_score=4.5))
            path = write_manifest(root/'source.jsonl', records)
            result = import_yo([path], root/'out', segmentation='complete-utterance', tail_padding_seconds=.2,
                               source_revision='test', log=lambda _: None)
            for row in load_manifest(result):
                self.assertEqual(row['duration'], 1.4)
                self.assertEqual(row['text'], 'Привет, мир!')

    def test_stop_average_can_hide_missed_terminal_and_metrics_are_per_utterance(self):
        p = torch.tensor([.001]*100+[.2])
        targets = torch.tensor([0.]*100+[1.])
        logits = torch.logit(p)
        self.assertLess(float(F.binary_cross_entropy_with_logits(logits, targets)), .02)
        result = stop_diagnostics(logits, targets, torch.zeros(101, dtype=torch.long), 1)
        self.assertEqual(float(result['stop_terminal_recall_05']), 0.)
        self.assertGreater(float(result['stop_terminal_bce']), 1.6)
        p = torch.tensor([.9]+[.01]*5+[.7,.9]+[.01]*20+[.4])
        targets = torch.tensor([0.]*7+[1.]+[0.]*20+[1.])
        owners = torch.tensor([0]*8+[1]*21)
        result = stop_diagnostics(torch.logit(p), targets, owners, 2)
        self.assertEqual(float(result['stop_terminal_recall_05']), .5)
        self.assertEqual(float(result['stop_terminal_recall_03']), 1.)
        self.assertEqual(float(result['stop_early_any_05']), .5)


def recognized(end=1.2, probability=.95):
    return [dict(word='Привет,', start=.1, end=.5, probability=probability),
            dict(word='мир!', start=.9, end=end, probability=probability)]


class AcousticDecisionTests(unittest.TestCase):
    def test_only_confident_complete_text_with_acoustic_tail_can_be_cropped(self):
        result = decide_tail('Привет, мир!', 2., recognized(), [(0.,1.2),(1.6,1.9)])
        self.assertEqual(result['action'], 'trim')
        self.assertEqual(result['cut_seconds'], 1.4)
        self.assertEqual(decide_tail('Привет, мир!', 2., recognized(), [(0.,1.3)])['action'], 'keep')
        self.assertEqual(decide_tail('Привет, мир!', 2., recognized(), [(0.,1.41)])['action'], 'keep')
        self.assertEqual(decide_tail('Привет, другой мир!', 2., recognized(), [(0.,1.9)])['action'], 'review')
        self.assertEqual(decide_tail('Привет, мир!', 2., recognized(probability=.6), [(0.,1.9)])['action'], 'review')
        self.assertEqual(decide_tail('Привет, мир!', 2., recognized(end=2.5), [(0.,1.9)])['action'], 'review')
        # An internal sentence/question boundary cannot satisfy the full text.
        self.assertEqual(decide_tail('Привет, мир! Как дела?', 2., recognized(), [(0.,1.9)])['action'], 'review')
        self.assertEqual(decide_tail('Привет, мир!', 2., recognized(), [(0.,1.9),(1.8,2.)])['action'], 'review')


class FreshPreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.raw = self.root/'raw'
        (self.raw/'data').mkdir(parents=True)
        self.expansion = self.root/'expanded'
        self.expansion.mkdir()
        self.serial = 0
        self.plan = dict(schema=1, repo=REPO, revision=REVISION, source_dir=str(self.raw),
                         selected=[], filters=dict(segmentation='complete-utterance'))
        groups, raw_records = {}, []
        for role in ('train','validation'):
            rows = []
            for i in range(2):
                raw_id = f'{role}-{i}'
                audio = self.audio(raw_id)
                end = 1.2 if i == 0 else 1.95
                raw_records.append(dict(utt_id=raw_id, local_spk_id=role, duration=2., text='привет мир',
                    text_denorm='Привет, мир!', text_alignment=f'привет(0.1,0.5) мир(0.9,{end}) [pause](%s,2.0)' % end,
                    audio={'bytes':b'not decoded in metadata pass', 'path':'ignored'}))
                rows.append(dict(utt_id=f'yo:{raw_id}:0', segment_source_id=f'yo:{raw_id}',
                    audio=str(audio), audio_sha256=file_digest(audio), text='Привет, мир!', language='Russian',
                    speaker_id='yo:'+role, source_id=role, source_revision=REVISION, start=0., end=2.))
            write_manifest(self.expansion/f'{role}.jsonl', rows)
            groups[role] = rows
        # Same speaker's utterances cross shard boundaries.
        for i, records in enumerate((raw_records[::2], raw_records[1::2])):
            path = self.raw/f'data/ru000-00{i}.parquet'
            pq.write_table(pa.Table.from_pylist(records), path)
            self.plan['selected'].append(dict(path=path.relative_to(self.raw).as_posix(),
                size=path.stat().st_size, sha256=file_digest(path)))
        self.plan_path = self.root/'plan.json'
        write_json(self.plan_path, self.plan)
        write_json(self.expansion/'expansion_state.json', dict(complete=True, plan_sha256=file_digest(self.plan_path)))
        write_json(self.expansion/'split_report.json', dict(plan_sha256=file_digest(self.plan_path),
            source_revision=REVISION, train_clips=2, validation_clips=2, train_hours=4/3600, validation_hours=4/3600))
        self.dialogs_paths = {}
        for role in ('train','validation'):
            rows = []
            for i in range(2):
                audio = self.audio(f'dialogs-{role}-{i}')
                rows.append(dict(utt_id=f'dialogs:{role}-{i}', audio=str(audio), audio_sha256=file_digest(audio),
                    text='Привет, мир!' if role == 'train' else 'Другой текст!', language='Russian',
                    speaker_id='dialogs:actor', source_id=f'dialogs:{role}',
                    segment_source_id=f'dialogs:{role}-{i}', start=0., end=2.))
            self.dialogs_paths[role] = write_manifest(self.root/f'dialogs-{role}.jsonl', rows)
        self.analyzer_calls = []
        calls = self.analyzer_calls
        class Analyzer:
            identity = {'fixture':'asr-vad'}
            def analyze(self, row):
                calls.append(row['utt_id'])
                # Train trims, validation is explicitly uncertain and unchanged.
                return dict(words=recognized(), speech_intervals=[(0.,1.2),(1.6,1.9)])
        self.analyzer = Analyzer()
        self.reviewed = self.root/'reviewed'
        self.review()
        self.recipe = json.loads((Path(__file__).resolve().parents[1]/'recipes/russian_trimmed_base_96gb.json').read_text())
        self.recipe['output_dir'] = str(self.root/'training-run')
        self.recipe['options']['replay_manifest'] = str(self.reviewed/'train.jsonl')
        self.recipe['options']['adaptation_validation_manifest'] = str(self.reviewed/'validation.jsonl')
        self.recipe_path = self.root/'recipe.json'
        write_json(self.recipe_path, self.recipe)

    def audio(self, name):
        self.serial += 1
        path = self.root/f'{name}.wav'
        sf.write(path, np.sin(np.arange(48000)*(.01+self.serial*.001))*.1, 24000, subtype='PCM_16')
        return path

    def review(self):
        return review_dialogs(self.dialogs_paths['train'], self.dialogs_paths['validation'],
                              self.reviewed, analyzer=self.analyzer, log=lambda _:None)

    def prepare(self, output=None):
        return prepare_retraining(self.expansion, self.plan_path, output or self.root/'new-data',
                                  recipe_path=self.recipe_path, log=lambda _:None)

    def test_audit_resume_and_uncertain_records_keep_original_audio(self):
        self.assertEqual(len(self.analyzer_calls), 4)
        report = self.review()
        self.assertEqual(len(self.analyzer_calls), 4)
        self.assertEqual(report['counts'], {'trim':2,'review':2})
        original = load_manifest(self.dialogs_paths['validation'])
        reviewed = load_manifest(self.reviewed/'validation.jsonl')
        self.assertEqual([r['audio_sha256'] for r in original], [r['audio_sha256'] for r in reviewed])
        trim = load_manifest(self.reviewed/'train.jsonl')[0]
        original = load_manifest(self.dialogs_paths['train'])[0]
        a, _ = sf.read(original['audio'], dtype='int16')
        b, _ = sf.read(trim['audio'], dtype='int16')
        np.testing.assert_array_equal(a[:33600], b)
        Path(trim['audio']).write_bytes(b'broken')
        with self.assertRaisesRegex(ValueError, 'Reviewed WAV changed'):
            self.review()

    def test_crop_preserves_text_split_and_samples_pins_clean_base_and_reuses_only_valid_cache(self):
        source_bytes = {p:p.read_bytes() for p in self.root.glob('*.wav')}
        report = self.prepare()
        root = self.root/'new-data'
        self.assertEqual(report['trimmed_clips'], 2)
        self.assertEqual(report['datasets']['train']['clips'], 2)
        config = json.loads((root/'recipe.json').read_text())
        self.assertIsNone(config['initial_adapter'])
        self.assertIsNone(config['resume_checkpoint'])
        TrainingOptions.parse(config['options']).check_base(config['options']['expected_base_fingerprint'])
        with self.assertRaisesRegex(ValueError, 'original unmerged'):
            TrainingOptions.parse(config['options']).check_base('0'*64)
        for role in ('train','validation'):
            old = load_manifest(self.expansion/f'{role}.jsonl')
            new = load_manifest(root/f'{role}.jsonl')
            self.assertEqual([r['utt_id'] for r in old], [r['utt_id'] for r in new])
            self.assertEqual([r['text'] for r in old], [r['text'] for r in new])
            for i,(a,b) in enumerate(zip(old,new)):
                before, _ = sf.read(a['audio'], dtype='int16')
                after, _ = sf.read(b['audio'], dtype='int16')
                np.testing.assert_array_equal(before[:len(after)], after)
                self.assertEqual(b['duration'], 1.4 if i == 0 else 2.)
                cache = FeatureCache.__new__(FeatureCache)
                cache.root = Path('/cache-fixture')
                self.assertEqual(cache.path(a) == cache.path(b), i == 1)
        self.assertEqual(source_bytes, {p:p.read_bytes() for p in source_bytes})
        with patch('fireredtts3_local.yo_retrain.metadata_records', side_effect=AssertionError('cached metadata')):
            self.assertEqual(self.prepare(), report)
        bad = load_manifest(root/'train.jsonl')[0]['audio']
        Path(bad).write_bytes(b'broken')
        with self.assertRaisesRegex(ValueError, 'Prepared audio/receipt changed'):
            self.prepare()

    def test_interrupted_preparation_resumes_and_input_changes_are_rejected(self):
        count = 0
        def interrupt(*args, **kwargs):
            nonlocal count
            count += 1
            if count == 3:
                raise KeyboardInterrupt()
            return crop_row(*args, **kwargs)
        with patch('fireredtts3_local.yo_retrain.crop_row', side_effect=interrupt):
            with self.assertRaises(KeyboardInterrupt):
                self.prepare()
        state = json.loads((self.root/'new-data'/'preparation_state.json').read_text())
        self.assertEqual(state['status'], 'interrupted')
        self.assertFalse((self.root/'new-data'/'recipe.json').exists())
        self.prepare()
        with self.assertRaisesRegex(ValueError, 'inputs/options changed'):
            prepare_retraining(self.expansion, self.plan_path, self.root/'new-data', padding=.3,
                               recipe_path=self.recipe_path, log=lambda _:None)

    def test_missing_parquet_stale_hash_and_old_adapter_fail_without_training_recipe(self):
        raw = self.raw/self.plan['selected'][0]['path']
        data = raw.read_bytes()
        raw.unlink()
        with self.assertRaisesRegex(ValueError, 'Parquet size/checksum'):
            self.prepare()
        raw.write_bytes(data[:-1]+bytes([data[-1]^1]))
        with self.assertRaisesRegex(ValueError, 'Parquet size/checksum'):
            self.prepare()
        self.assertFalse((self.root/'new-data'/'recipe.json').exists())
        self.recipe['initial_adapter'] = 'old-adapter'
        write_json(self.recipe_path, self.recipe)
        with self.assertRaisesRegex(ValueError, 'initial adapter'):
            self.prepare(self.root/'another')


if __name__ == '__main__':
    unittest.main()

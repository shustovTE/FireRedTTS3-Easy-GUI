"""Whole-utterance import, historical holdouts and bounded-memory coverage."""
from dataclasses import replace
import json
from pathlib import Path
import random
import signal
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf
import torch

from fireredtts3_local.training_data import (SpeakerPairs, assert_disjoint,
    load_manifest, split_with_anchors)
from fireredtts3_local.training_run import TrainingOptions, bounded_batches, planned_batches
from fireredtts3_local.yo_dataset import aligned_segments, has_sentence_ending, import_yo
import test_russian_training as fixtures


class CompleteUtteranceTests(unittest.TestCase):
    def test_endings_quotes_and_ellipsis(self):
        for text in ['Готово.', 'Готово!', 'Готово?', 'Готово?!', 'Готово!!',
                     'Он сказал: «Готово!»', 'Он сказал: «Готово».', '«Готово.»  ',
                     'Сделано (уже!).', 'Сначала… А теперь готово.']:
            with self.subTest(text=text):
                self.assertTrue(has_sentence_ending(text))
        for text in ['', 'Готово', 'Готово,', 'Готово:', 'Готово;', 'Готово…',
                     'Готово...', 'Готово..', 'Готово!…', 'Готово!..', 'Готово. . .',
                     '«Готово…»', 'Готово...? ', 'Готово⋯', 'Готово…?»']:
            with self.subTest(text=text):
                self.assertFalse(has_sentence_ending(text))

    def test_whole_text_and_alignment_checks(self):
        row = fixtures.ImportTests().row()
        self.assertEqual(aligned_segments(row, max_seconds=2, segmentation='complete-utterance'),
                         [(0., 4., 'Привет, мир. Это тест!')])
        for ending in ['тест', 'тест,', 'тест...', 'тест…']:
            bad = dict(row, text_denorm='Привет, мир. Это ' + ending)
            with self.assertRaisesRegex(ValueError, 'incomplete_sentence'):
                aligned_segments(bad, segmentation='complete-utterance')
        with self.assertRaisesRegex(ValueError, 'denormalized_transcript_mismatch'):
            aligned_segments(dict(row, text_denorm='Привет, мир. Это другой тест!'), segmentation='complete-utterance')
        with self.assertRaisesRegex(ValueError, 'alignment_outside_audio'):
            aligned_segments(dict(row, duration=3.), segmentation='complete-utterance')

    def test_short_and_over_120_second_audio_stays_whole(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = []
            for i, duration in enumerate((.4, 121.)):
                signal = (np.sin(np.arange(round(duration*24000))*.01)*.1).astype('float32')
                sf.write(root / f'{i}.wav', signal, 24000)
                rows.append(dict(utt_id=str(i), audio=f'{i}.wav', text='привет мир', text_denorm='Привет, мир!',
                                 text_alignment=f'привет(0.01,{duration/2:.3f}) мир({duration/2:.3f},{duration-.01:.3f})',
                                 duration=duration, local_spk_id='video__spk0', lang='ru', mos_score=4.5))
            rows += [dict(rows[0], utt_id='bad-punctuation', text_denorm='Привет, мир…'),
                     dict(rows[0], utt_id='bad-mos', mos_score=3.9),
                     dict(rows[0], utt_id='bad-language', lang='en'),
                     dict(rows[0], utt_id='singleton', local_spk_id='other__spk0')]
            source = root / 'source.jsonl'
            source.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')
            result = import_yo([source], root/'out', max_hours=0, segmentation='complete-utterance',
                               source_revision='test', log=lambda _: None)
            actual = load_manifest(result)
            self.assertEqual([r['duration'] for r in actual], [.4, 121.])
            self.assertEqual([r['text'] for r in actual], ['Привет, мир!']*2)
            for i, row in enumerate(actual):
                original, _ = sf.read(root/f'{i}.wav')
                saved, _ = sf.read(row['audio'])
                np.testing.assert_array_equal(original, saved)
            pairs = SpeakerPairs(actual, None, None)
            prompt, target = pairs.sample(random.Random(4))
            self.assertNotEqual(prompt['audio_sha256'], target['audio_sha256'])
            report = json.loads((root/'out'/'import_report.json').read_text())
            self.assertIsNone(report['max_segment_seconds'])
            self.assertEqual(report['rejections']['incomplete_sentence_ending'], 1)
            self.assertEqual(report['rejections']['language_or_quality'], 2)
            self.assertEqual(report['rejections']['segments_without_distinct_reference'], 1)


def row(speaker, clip, **extra):
    return dict(utt_id=f'{speaker}-{clip}', audio_sha256=f'{speaker}-{clip}', text='Готово.',
                language='Russian', duration=40., speaker_id=speaker, source_id=speaker, **extra)


class AnchoredSplitTests(unittest.TestCase):
    def test_resegmentation_preserves_old_speakers_and_unseen_groups(self):
        old_train, old_val = [row('a', 0)], [row('b', 0)]
        new = [row(speaker, clip) for speaker in 'abcd' for clip in (1, 2)]
        train, val = split_with_anchors(new, old_train, old_val, .4, 12)
        self.assertTrue(all(r in train for r in new if r['speaker_id'] == 'a'))
        self.assertTrue(all(r in val for r in new if r['speaker_id'] == 'b'))
        self.assertCountEqual(train + val, new)
        assert_disjoint(train + old_train, val + old_val)
        SpeakerPairs(train, None, None)
        SpeakerPairs(val, None, None)
        self.assertEqual((train, val), split_with_anchors(new, old_train, old_val, .4, 12))

    def test_missing_old_rows_still_connect_new_global_identity(self):
        old_train = [row('a', 0, global_spk_id='person'), row('a', 1)]
        old_val = [row('b', 0)]
        new = [row('new-name', 1, global_spk_id='person'), row('a', 4), row('b', 2)]
        train, val = split_with_anchors(new, old_train, old_val)
        self.assertCountEqual(train, new[:2])
        self.assertEqual(val, new[2:])
        # New information connecting historically separate splits cannot be
        # solved by moving a trained identity into validation.
        new[0]['source_id'] = 'b'
        with self.assertRaisesRegex(ValueError, 'bridge'):
            split_with_anchors(new, old_train, old_val)


class LongBatchTests(unittest.TestCase):
    def test_unlimited_is_explicit_and_invalid_limits_still_fail(self):
        TrainingOptions.parse({'max_prompt_seconds': None, 'max_target_seconds': None})
        for value in (0, -1, True, 'unlimited', float('inf')):
            with self.subTest(value=value), self.assertRaises(ValueError):
                TrainingOptions.parse({'max_target_seconds': value})
        long = [row('a', 1), row('a', 2)]
        with self.assertRaisesRegex(ValueError, 'limit'):
            SpeakerPairs(long)
        SpeakerPairs(long, None, None)

    def test_budget_splits_batches_never_skips_oversized_pair(self):
        pairs = [(row('a', 1), row('a', 2), .1)]*4
        batches = list(bounded_batches(pairs, 8, 1100))
        self.assertEqual([len(b) for b in batches], [2, 2])
        oversized = [(dict(pairs[0][0], duration=500), pairs[0][1], .1)]
        self.assertEqual(list(bounded_batches(oversized, 8, 100)), [oversized])

    def test_dynamic_coverage_and_resume_keep_all_targets_and_update_boundaries(self):
        rows = [row(s, i) for s in 'abc' for i in range(3)]
        pairs = SpeakerPairs(rows, None, None)
        opts = TrainingOptions(max_prompt_seconds=None, max_target_seconds=None,
                               sampling_strategy='coverage', batch_size=4, batch_token_budget=600)
        state = random.Random(9).getstate()
        batches = list(planned_batches(pairs, None, opts, state, 0, 3, 1, 'Epochs'))
        self.assertEqual([len(b.pairs) for b in batches], [1]*9)
        self.assertEqual([i for i,b in enumerate(batches) if b.sampler_state], [3, 7, 8])
        targets = [pair[1]['utt_id'] for b in batches for pair in b.pairs]
        self.assertCountEqual(targets, [r['utt_id'] for r in rows])
        resumed = list(planned_batches(pairs, None, opts, batches[3].sampler_state, 1, 3, 1, 'Epochs'))
        self.assertEqual(resumed, batches[4:])
        plain = list(planned_batches(pairs, None, replace(opts, batch_token_budget=0), state, 0, 3, 1, 'Epochs'))
        self.assertEqual([pair for b in plain for pair in b.pairs], [pair for b in batches for pair in b.pairs])

    def test_cli_sigterm_finishes_update_and_saves_resume_state(self):
        from fireredtts3_local import adapt, training_run
        fixture = fixtures.DataTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        output = fixture.root/'signal-run'
        recipe = fixture.root/'recipe.json'
        recipe.write_text(json.dumps(dict(manifest_path=str(fixture.manifest), output_dir=str(output),
            rank=2, alpha=4, training_steps=3, gradient_accumulation=1,
            options=dict(validation_pairs=2, validate_every=10, gradient_checkpointing=False,
                         cache_dir='', patches_per_example='all', dit_chunk_size=2))))
        original = training_run.batch_flow_loss
        def stop_in_update(tts, *args, **kwargs):
            result = original(tts, *args, **kwargs)
            if tts.tts_core.training:
                signal.raise_signal(signal.SIGTERM)
            return result
        previous_handler = signal.getsignal(signal.SIGTERM)
        with patch.object(sys, 'argv', ['adapt', 'train', str(recipe)]), \
             patch('fireredtts3_local.model_manager.ModelManager.load_base', fixtures.Manager.load_base), \
             patch('fireredtts3_local.model_manager.ModelManager.unload', fixtures.Manager.unload), \
             patch.object(training_run, 'require_training_hardware'), \
             patch.object(training_run, 'model_identity', return_value='toy'), \
             patch.object(training_run, 'batch_flow_loss', side_effect=stop_in_update):
            adapt.main()
        state = torch.load(output/'trainer_state.pt', weights_only=True)
        self.assertEqual(state['step'], 1)
        self.assertEqual(state['scheduler']['last_epoch'], 1)
        self.assertTrue((output/'adapter_model.safetensors').is_file())
        self.assertEqual(signal.getsignal(signal.SIGTERM), previous_handler)

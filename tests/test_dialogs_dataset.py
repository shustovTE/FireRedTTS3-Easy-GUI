"""Selection quotas, extra session holdout, source integrity and real WAV audit."""
import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

from fireredtts3_local.dialogs_dataset import (audio_name, category, choose_thirds,
    eligible, file_inventory, lexical_key, plan_dialogs, prepare_dialogs, short_support_rows,
    validate_plan, METADATA, REPO, REVISION)
from fireredtts3_local.training_data import SpeakerPairs, assert_disjoint, load_manifest, write_manifest


class DialogsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root/'source'
        (self.source/'wavs').mkdir(parents=True)
        rows = []
        for speaker in 'DMS':
            # Enough independent sessions to reserve validation without taking
            # away the 30-second training budget, even when random slots use
            # extra questions/statements from a small session.
            for session in range(6):
                for index in range(12):
                    label = f'{speaker} {session} {index}'
                    text = [f'Почему {label}? Спасибо.', f'Вот ответ {label}.', f'Отлично {label}!'][index%3]
                    name = f'wavs/recording_{speaker.lower()}_{session}_{index}.wav'
                    duration = 1.2 + .1*(index%6)
                    signal = np.sin(np.arange(round(24000*duration))*(.02+len(rows)*.0001))*.1
                    sf.write(self.source/name, signal, 24000)
                    rows.append(dict(audio_path=name, speaker_id=speaker, text=text, emotion='neutral',
                                     emotion_on_russian='нейтрально', accent_text=text, duration=str(duration)))
        self.rows = rows
        self.write_sources(rows[:-2], rows[-2:-1], rows[-1:])
        self.write_inventory()

    def write_sources(self, train, val, test):
        for name, rows in [('metadata.csv', train+val+test), ('train.csv', train), ('val.csv', val), ('test.csv', test)]:
            with (self.source/name).open('w', encoding='utf-8', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(self.rows[0]), delimiter='|')
                writer.writeheader(); writer.writerows(rows)
        (self.source/'LICENSE.md').write_text('test license')

    def write_inventory(self, excluded=()):
        path = self.source/f'file-inventory-{REVISION}.json'
        files = sorted(set(METADATA) | {r['audio_path'] for r in self.rows if r['audio_path'] not in excluded})
        path.write_text(json.dumps(dict(schema=1, repo=REPO, revision=REVISION, files=files), indent=2))
        return path

    def plan(self, folder='plan'):
        report = plan_dialogs(self.source, self.root/folder, max_seconds=3, train_minutes=.5,
                              validation_minutes=.2, seed=1234)
        return self.root/folder/'plan.json', report

    def test_neutral_duration_endings_and_question_anywhere(self):
        pool = [dict(self.rows[0], text='Почему? Всё готово.'),
                dict(self.rows[1], text='Готово.'), dict(self.rows[2], text='Готово!'),
                dict(self.rows[3], emotion='happy'), dict(self.rows[4], duration='10.001'),
                dict(self.rows[5], text='Готово…')]
        actual, rejected = eligible(pool)
        self.assertEqual([r['category'] for r in actual], ['question', 'statement', 'other'])
        self.assertEqual(rejected, {'not_neutral':1, 'duration':1, 'incomplete_sentence_ending':1})
        self.assertEqual(category('Правда? Да.'), 'question')

    def test_plan_reproducible_extra_validation_and_no_text_or_session_leaks(self):
        path, report = self.plan()
        other, report2 = self.plan('other')
        self.assertEqual(report, report2)
        a, b = json.loads(path.read_text()), json.loads(other.read_text())
        self.assertEqual(a, b)
        validate_plan(a)
        official_test = {r['audio_path'] for r in self.rows[-2:]}
        self.assertFalse(official_test & {r['audio_path'] for r in a['train']+a['validation']})
        self.assertFalse({r['session'] for r in a['train']} & {r['session'] for r in a['validation']})
        self.assertFalse({lexical_key(r['text']) for r in a['train']} & {lexical_key(r['text']) for r in a['validation']})
        for speaker in 'DMS':
            for split, budget in [('train', 30), ('validation',12)]:
                summary = report['speakers'][speaker][split]
                self.assertGreaterEqual(summary['minutes']*60, budget-3)
                self.assertLessEqual(summary['minutes']*60, budget+1e-6)
                self.assertEqual(set(summary['slots'].values()), {summary['clips']//3})

    def test_random_third_can_reuse_types_but_not_files_or_texts(self):
        rows, _ = eligible([r for r in self.rows if r['speaker_id']=='D'])
        # No exclamatory leftovers: random third MUST still be fillable.
        rows = [r for r in rows if r['category'] in {'question','statement'}]
        rows += [dict(rows[0], audio_path='wavs/duplicate_999.wav')]
        selected = choose_thirds(rows, 30, 'fixture')
        self.assertTrue(selected)
        self.assertEqual(len(selected), len({r['audio_path'] for r in selected}))
        self.assertEqual(len(selected), len({r['text'] for r in selected}))
        self.assertTrue(all(r['category'] in {'question','statement'} for r in selected if r['selection_slot']=='random'))

    def test_missing_remote_wavs_reselected_before_quotas_and_holdout(self):
        path, _ = self.plan()
        old = json.loads(path.read_text())
        missing = {next(r['audio_path'] for r in old[split] if r['speaker_id']==speaker)
                   for split in ('train','validation') for speaker in 'DMS'}
        self.write_inventory(missing)
        # A stray local WAV does not make an absent remote entry eligible.
        self.assertTrue(all((self.source/name).is_file() for name in missing))
        new_path, report = self.plan('replanned')
        new = json.loads(new_path.read_text())
        validate_plan(new)
        self.assertFalse(missing & {r['audio_path'] for r in new['train']+new['validation']})
        self.assertEqual(report['official_train_rejections']['missing_audio'],len(missing))
        self.assertEqual(report['file_inventory']['metadata_missing_audio'],len(missing))
        for speaker in 'DMS':
            for split, budget in [('train',30),('validation',12)]:
                stats=report['speakers'][speaker][split]
                self.assertGreaterEqual(stats['minutes']*60,budget-3)
                self.assertLessEqual(stats['minutes']*60,budget+1e-6)
                self.assertEqual(set(stats['slots'].values()),{stats['clips']//3})

    def test_old_plan_rejected_before_any_audio_download_or_output(self):
        path, _ = self.plan()
        legacy = json.loads(path.read_text())
        del legacy['file_inventory_sha256']
        del legacy['report']['file_inventory']
        path.write_text(json.dumps(legacy))
        files=json.loads(self.write_inventory({legacy['train'][0]['audio_path']}).read_text())['files']
        with patch('huggingface_hub.HfApi') as api, patch('fireredtts3_local.dialogs_dataset.fetch_files') as fetch:
            api.return_value.list_repo_files.return_value=files
            with self.assertRaisesRegex(ValueError,'1 WAVs absent.*Rerun plan-dialogs'):
                prepare_dialogs(path,self.root/'import',download=True)
            fetch.assert_not_called()
            api.return_value.list_repo_files.assert_called_once_with(REPO,repo_type='dataset',revision=REVISION)
        self.assertFalse((self.root/'import').exists())

    def test_inventory_download_offline_integrity_and_network_errors(self):
        path=self.write_inventory()
        files=json.loads(path.read_text())['files']
        with patch('huggingface_hub.HfApi') as api, patch('fireredtts3_local.dialogs_dataset.fetch_files'):
            api.return_value.list_repo_files.return_value=files
            report=plan_dialogs(self.source,self.root/'online',max_seconds=3,train_minutes=.5,
                                validation_minutes=.2,download=True)
            api.return_value.list_repo_files.assert_called_once_with(REPO,repo_type='dataset',revision=REVISION)
        _, offline_report=self.plan('offline')
        self.assertEqual(report,offline_report)
        previous=path.read_bytes()
        with patch('huggingface_hub.HfApi') as api:
            api.return_value.list_repo_files.side_effect=RuntimeError('network unavailable')
            with self.assertRaisesRegex(RuntimeError,'network unavailable'):
                file_inventory(self.source,REVISION,refresh=True)
        self.assertEqual(path.read_bytes(),previous)
        with patch('huggingface_hub.HfApi') as api:
            api.return_value.list_repo_files.return_value=[]
            with self.assertRaisesRegex(ValueError,'Invalid pinned file inventory'):
                file_inventory(self.source,REVISION,refresh=True)
        self.assertEqual(path.read_bytes(),previous)
        path.write_bytes(previous+b'\n')
        with self.assertRaisesRegex(ValueError,'File inventory changed'):
            prepare_dialogs(self.root/'online'/'plan.json',self.root/'bad')
        invalid=json.loads(path.read_text()); invalid['revision']='0'*40
        path.write_text(json.dumps(invalid))
        with self.assertRaisesRegex(ValueError,'Invalid pinned file inventory'):
            file_inventory(self.source,REVISION)
        path.unlink()
        with self.assertRaisesRegex(ValueError,'Missing pinned file inventory'):
            self.plan('missing')

    def test_import_real_audio_and_drop_whole_bad_triple(self):
        path, _ = self.plan()
        plan = json.loads(path.read_text())
        broken = plan['train'][0]
        # Existing metadata remains intact; actual audio is now clipped.
        sf.write(self.source/broken['audio_path'], np.ones(round(broken['duration']*24000)), 24000)
        report = prepare_dialogs(path, self.root/'import', log=lambda _:None)
        train = load_manifest(self.root/'import'/'train.jsonl')
        val = load_manifest(self.root/'import'/'validation.jsonl')
        self.assertEqual(len(train), len(plan['train'])-3)
        self.assertEqual(len(val), len(plan['validation']))
        self.assertEqual(report['audio_rejections'][broken['audio_path']], 'clipping')
        self.assertEqual(len(report['discarded_triple_members']), 3)
        assert_disjoint(train, val, 'recording')
        SpeakerPairs(train,3,3); SpeakerPairs(val,3,3)
        for row in train+val:
            self.assertEqual(sf.info(row['audio']).samplerate,24000)
            self.assertLessEqual(row['duration'],3)
        for split in ('train','validation'):
            for speaker in 'DMS':
                summary=report['splits'][split][speaker]
                self.assertEqual(set(summary['slots'].values()), {summary['clips']//3})

    def test_tampered_metadata_plan_and_paths_rejected(self):
        for path in ('../secret.wav','/tmp/a.wav','wavs/../a.wav','wavs/a\\b.wav'):
            with self.subTest(path=path), self.assertRaises(ValueError): audio_name(path)
        path, _ = self.plan()
        plan=json.loads(path.read_text())
        altered=copy.deepcopy(plan)
        altered['train'][0]['selection_slot']='statement'
        with self.assertRaisesRegex(ValueError,'punctuation'): validate_plan(altered)
        altered=copy.deepcopy(plan)
        altered['train'][0]['text']='Подменённый текст?'
        path.write_text(json.dumps(altered))
        with self.assertRaisesRegex(ValueError,'official eligible'): prepare_dialogs(path,self.root/'bad')
        path.write_text(json.dumps(plan))
        with (self.source/'train.csv').open('a') as stream: stream.write('\n')
        with self.assertRaisesRegex(ValueError,'metadata changed'): prepare_dialogs(path,self.root/'bad')

    def test_official_split_overlap_rejected(self):
        self.write_sources(self.rows, self.rows[:1], [])
        with self.assertRaisesRegex(ValueError,'Duplicate audio|split overlap'):
            self.plan()

    def test_short_yo_support_keeps_distinct_pairs_and_existing_split(self):
        manifest_rows=[]
        for i,row in enumerate(self.rows[:4]):
            manifest_rows.append(dict(audio=str(self.source/row['audio_path']), text='Готово.' if i!=2 else 'Не закончено,',
                speaker_id='paired' if i<3 else 'singleton', language='Russian'))
        path=write_manifest(self.root/'yo.jsonl',manifest_rows)
        selected, report=short_support_rows(path,10)
        self.assertEqual(len(selected),2)
        self.assertEqual(report['speakers'],1)
        SpeakerPairs(selected,10,10)

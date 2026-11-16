"""CPU native rollouts, lossless sentence assembly and resumable ASR scoring."""
import copy
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import soundfile as sf

from fireredtts3_local import cfg_diagnostics as shared
from fireredtts3_local import sentence_diagnostics as diag
from fireredtts3_local.training_data import file_digest
from fireredtts3_local.training_monitor import load_benchmark
import test_cfg_diagnostics as fixtures


class SentenceTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.DiagnosticTests()
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.setUp()
        f = self.fixture
        rows = [json.loads(line) for line in f.benchmark.read_text().splitlines()]
        rows[0]['text'] = 'Первое предложение. Второе предложение?'
        rows[1]['text'] = 'Короткая фраза! Последняя фраза.'
        f.benchmark.write_text(''.join(json.dumps(row)+'\n' for row in rows))
        _, fingerprint = load_benchmark(f.benchmark)
        f.recipe['generation_benchmark_fingerprint'] = fingerprint
        (f.run/'recipe.json').write_text(json.dumps(f.recipe))
        for path in f.run.glob('*/firered_lora_metadata.json'):
            value = json.loads(path.read_text())
            value['recipe'] = f.recipe
            path.write_text(json.dumps(value))
        previous = f.run/'step0.json'
        value = json.loads(previous.read_text())
        value['fingerprint'] = fingerprint
        previous.write_text(json.dumps(value))

    def plan(self):
        return diag.make_plan(self.fixture.run, self.fixture.output)

    def generate(self, plan):
        return shared.execute(plan, manager=fixtures.FakeManager(self.fixture.original), log=lambda _: None)

    @staticmethod
    def fake_asr(fail_at=None):
        calls = []
        def transcribe(audio, **settings):
            calls.append(audio)
            if settings != diag.ASR_SETTINGS:
                raise AssertionError('Unexpected ASR settings or text prompt')
            if len(calls) == fail_at:
                raise RuntimeError('ASR interrupted')
            segment = SimpleNamespace(start=0., end=.5, text='Первое предложение.',
                avg_logprob=-.2, no_speech_prob=.01,
                words=[SimpleNamespace(start=0., end=.2, word='Первое', probability=.9),
                       SimpleNamespace(start=.2, end=.5, word='предложение.', probability=.9)])
            return iter([segment]), None
        return SimpleNamespace(transcribe=transcribe), calls

    def test_split_preserves_words_decimals_quotes_initials_and_abbreviations(self):
        text = 'А. С. Пушкин жил в г. Москве. Цена 3.14 рубля! «Вы готовы?» Да, по ул. Мира.'
        parts = diag.split_sentences(text)
        self.assertEqual(parts, ['А. С. Пушкин жил в г. Москве.', 'Цена 3.14 рубля!',
                                 '«Вы готовы?»', 'Да, по ул. Мира.'])
        self.assertEqual(' '.join(parts), text)
        self.assertEqual(diag.split_sentences('Первая строка.\n\nВторая строка?'),
                         ['Первая строка.', 'Вторая строка?'])
        for text in ('', 'Незаконченная реплика'):
            with self.assertRaisesRegex(ValueError, 'sentence punctuation'):
                diag.split_sentences(text)

    def test_wer_deletions_insertions_substitutions_and_missing_spans(self):
        result = diag.transcript_errors('Раз два три четыре пять шесть.', 'Раз два пять шесть.')
        self.assertEqual(result['wer'], 2/6)
        self.assertEqual(result['deletion'], 2)
        self.assertEqual(result['substitution'], 0)
        self.assertEqual(result['insertion'], 0)
        self.assertEqual(result['missing_spans'], [dict(start_word=2, end_word=4, text='три четыре', words=2)])
        self.assertEqual(diag.transcript_errors('Первое слово.', 'Первое другое.')['substitution'], 1)
        self.assertEqual(diag.transcript_errors('Первое слово.', 'Первое лишнее слово.')['insertion'], 1)
        self.assertEqual(diag.transcript_errors('Ёлка, ёж!', 'елка еж')['wer'], 0)
        self.assertEqual(diag.transcript_errors('Пустой ответ.', '')['deletion'], 2)
        self.assertGreater(diag.transcript_errors('Раз.', 'Раз два три четыре')['wer'], 1)

    def test_plan_keeps_reference_seed_and_exact_component_selection(self):
        plan = self.plan()
        self.assertEqual(plan['total_cases'], 18)
        self.assertEqual(plan['total_comparisons'], 12)
        self.assertEqual([(s['name'], s['active_components']) for s in plan['stages']],
                         [('base', []), ('backbone_only', ['backbone']), ('dit_only', ['dit'])])
        self.assertEqual(plan['source_step'], 143)
        for row in plan['cases']:
            original = next(r for r in plan['original_cases'] if r['id'] == row['source_case_id'])
            for field in ('reference_audio', 'reference_text', 'reference_sha256', 'seed', 'language'):
                self.assertEqual(row[field], original[field])
        shared.verify_plan(plan)
        changed = copy.deepcopy(plan)
        changed['asr_settings']['vad_filter'] = True
        with self.assertRaisesRegex(ValueError, 'runtime changed'):
            shared.verify_plan(changed)
        previous = self.fixture.run/'step0.json'
        data = json.loads(previous.read_text())
        data['results'].pop()
        previous.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, 'base report'):
            self.plan()

    def test_native_matrix_assembly_and_report_packaging(self):
        plan = self.plan()
        f = self.fixture
        original_hashes = {p: file_digest(p) for p in f.run.rglob('*') if p.is_file()}
        generation = self.generate(plan)
        self.assertTrue(generation['complete'])
        for row in generation['results']:
            is_control = row['stage'] == 'base' and row['mode'] == 'whole'
            self.assertEqual(row['previous_monitor'] is not None, is_control)
            if is_control:
                self.assertFalse(row['previous_monitor']['reproduction_review_required'])
        self.assertFalse(list(f.output.glob('diagnostics-*.zip')))
        recognizer, calls = self.fake_asr()
        with patch.object(diag, 'load_asr', return_value=(recognizer, {'test_model_hash': 'fixed'})):
            result = diag.score(plan, log=lambda _: None)
        self.assertEqual(len(calls), 12)
        self.assertTrue(result['complete'])
        self.assertTrue(result['reproduction_controls_complete'])
        self.assertFalse(result['reproduction_review_required'])
        for row in result['results']:
            parts = [r for r in generation['results'] if r['stage'] == row['stage']
                     and r['source_case_id'] == row['id'] and r['mode'] == row['mode']]
            arrays = [sf.read(f.output/r['float_audio'], dtype='float32')[0] for r in parts]
            expected = arrays[0] if row['mode'] == 'whole' else np.concatenate(
                [arrays[0], np.zeros(200, dtype=np.float32), arrays[1]])
            actual, rate = sf.read(f.output/row['float_audio'], dtype='float32')
            np.testing.assert_array_equal(actual, expected)
            self.assertEqual(rate, 1000)
            self.assertEqual(row['added_silence_seconds'], .2 if row['mode'] == 'sentences' else 0.)
            self.assertEqual(row['parts_count'], len(parts))
            self.assertIsNotNone(row['wer_delta_vs_base_same_mode'])
            self.assertIsNotNone(row['wer_delta_vs_whole_same_weights'])
        with zipfile.ZipFile(f.output/'diagnostics-report.zip') as archive:
            self.assertIn('comparison.json', archive.namelist())
            self.assertNotIn('plan.json', archive.namelist())
        names = []
        for path in f.output.glob('diagnostics-audio-*.zip'):
            with zipfile.ZipFile(path) as archive:
                names.extend(archive.namelist())
        self.assertEqual(len(names), 12)
        self.assertTrue(all(name.startswith('comparison-audio/') and '.float.wav' not in name for name in names))
        for path, digest in original_hashes.items():
            self.assertEqual(file_digest(path), digest)

    def test_asr_resume_keeps_finished_wavs_and_rejects_changed_inputs(self):
        plan = self.plan()
        self.generate(plan)
        f = self.fixture
        recognizer, calls = self.fake_asr(fail_at=3)
        identity = {'test_model_hash': 'fixed'}
        with patch.object(diag, 'load_asr', return_value=(recognizer, identity)):
            with self.assertRaisesRegex(RuntimeError, 'ASR interrupted'):
                diag.score(plan, log=lambda _: None)
        partial = json.loads((f.output/'comparison.json').read_text())
        self.assertEqual(partial['completed_cases'], 2)
        retained = {name: file_digest(f.output/name) for row in partial['results'] for name in row['artifacts']}
        recognizer, calls = self.fake_asr()
        with patch.object(diag, 'load_asr', return_value=(recognizer, identity)):
            final = diag.score(plan, log=lambda _: None)
        self.assertTrue(final['complete'])
        self.assertEqual(len(calls), 10)
        for name, digest in retained.items():
            self.assertEqual(file_digest(f.output/name), digest)
        with patch.object(diag, 'load_asr', return_value=(recognizer, {'test_model_hash': 'changed'})):
            with self.assertRaisesRegex(ValueError, 'ASR model changed'):
                diag.score(plan, log=lambda _: None)
        altered = f.output/next(iter(retained))
        saved = altered.read_bytes()
        altered.write_bytes(b'changed')
        with patch.object(diag, 'load_asr', return_value=(recognizer, identity)):
            with self.assertRaisesRegex(ValueError, 'audio changed'):
                diag.score(plan, log=lambda _: None)
        altered.write_bytes(saved)
        generation = json.loads((f.output/'report.json').read_text())
        raw = f.output/generation['results'][0]['float_audio']
        raw.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'audio changed'):
            diag.score(plan, log=lambda _: None)

    def test_asr_weights_pinned_and_loaded_without_gpu_or_network_on_resume(self):
        plan = self.plan()
        path = self.fixture.root/'asr-model'
        path.mkdir()
        for name in ('model.bin', 'config.json', 'tokenizer.json'):
            (path/name).write_bytes(b'fake model')
        plan['asr_model'] = str(path)
        self.fixture.output.mkdir()
        with patch('faster_whisper.WhisperModel') as model, patch('faster_whisper.utils.download_model') as download:
            _, identity = diag.load_asr(plan, self.fixture.output, lambda _: None)
            model.assert_called_once_with(str(path), device='cpu', compute_type='int8',
                                          cpu_threads=4, num_workers=1, local_files_only=True)
            _, resumed = diag.load_asr(plan, self.fixture.output, lambda _: None)
            self.assertEqual(identity, resumed)
            download.assert_not_called()
            (path/'model.bin').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'ASR weights/runtime changed'):
                diag.load_asr(plan, self.fixture.output, lambda _: None)

    def test_sentence_worker_runs_asr_after_generation_and_propagates_failure(self):
        job = self.fixture.root/'job'
        job.mkdir()
        state = dict(cwd=str(Path.cwd()), resume=True)
        (job/'status.json').write_text(json.dumps(state))
        (job/'plan.json').write_text(json.dumps(self.plan()))
        fd = os.open(self.fixture.root/'worker.lock', os.O_CREAT | os.O_WRONLY, 0o600)
        with patch('sys.argv', ['cfg_diagnostics', '_worker', str(job), str(fd)]), \
                patch.object(shared, 'run_commands', return_value=17) as run:
            self.assertEqual(shared.main(), 17)
        commands = run.call_args.args[2]
        self.assertEqual([phase for phase, _ in commands], ['diagnostics', 'asr'])
        self.assertIn('--resume', commands[0][1])
        self.assertIn('fireredtts3_local.sentence_diagnostics', commands[1][1])
        self.assertEqual(commands[1][1][-2], '_score')

    def test_cli_selects_sentence_defaults(self):
        with patch('sys.argv', ['sentence_diagnostics', 'start']), \
                patch.object(shared, 'launch', return_value={}) as launch, patch('sys.stdout', new_callable=io.StringIO):
            self.assertEqual(diag.main(), 0)
        self.assertEqual(launch.call_args.kwargs['output'], diag.DEFAULT_OUTPUT)
        self.assertEqual(launch.call_args.kwargs['job_dir'], diag.DEFAULT_JOB)
        self.assertEqual(launch.call_args.kwargs['sentence_options'],
                         dict(step=143, asr_model='large-v3', pause_seconds=.2))


if __name__ == '__main__':
    unittest.main()

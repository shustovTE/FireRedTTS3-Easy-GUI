import csv
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import soundfile as sf
import torch

from fireredtts3_local import vox_attention_ablation as ablate
from fireredtts3_local.vox_recipe_transfer import configure
from test_training_batch import tiny_core


class AblationTests(unittest.TestCase):
    def test_complete_backbone_suite_uses_historical_voices_seeds_and_recovers_missing_wav(self):
        with tempfile.TemporaryDirectory() as tmp:
            step = Path(tmp)/'step-0250'
            common = dict(checkpoints={'step-0250': {'path': 'checkpoint'}}, base_sha256=ablate.BASE_SHA,
                          settings=ablate.SETTINGS, runtime={})
            def series(folder, *, base_fingerprint):
                self.assertEqual(base_fingerprint, ablate.BASE_SHA)
                groups = [('question', (1234, 1235)), ('cities', (1234, 1235)), ('timofey', (1234, 1235))]
                if folder.name == 'series-2':
                    groups = [('cities', (1236, 1237, 1238, 1239))]
                cases = [dict(id=f'{name}_seed{seed}', text=name, seed=seed, language='Russian',
                              reference_text=f'Референс {seed}', reference_audio=f'{seed}.wav',
                              reference_sha256=str(seed)) for name, seeds in groups for seed in seeds]
                return dict(common, cases=cases, series=str(folder), benchmark_sha256=folder.name)
            pack = step/'backbone-complete'
            generated = (np.ones(2400, dtype=np.float32)*1.1, 24000, {'finish_reason':'stop'})
            with (patch.object(ablate, 'model_identity', return_value=ablate.BASE_SHA),
                  patch.object(ablate, 'series_plan', side_effect=series),
                  patch.object(ablate, 'ModelManager'), patch.object(ablate, 'route', return_value={'dit': 0}) as route,
                  patch.object(ablate, 'generate_case', return_value=generated) as gen,
                  patch('sys.argv', ['ablation', '--full-backbone', str(step), '--audio-only'])):
                ablate.main()
                report = json.loads((pack/'report.json').read_text())
                self.assertTrue(report['audio_complete'])
                self.assertEqual(report['plan']['variants'], ['backbone'])
                self.assertFalse(report['plan']['settings']['do_split'])
                self.assertEqual(len(report['results']), 58)
                self.assertEqual(route.call_args.args[1], 'backbone')
                for seed in range(1234, 1240):
                    self.assertIn(f'controls/cities_seed{seed}.wav', report['results'])
                for asset, seeds, ref in [('listening-ru-500', (1234, 1235), '1237.wav'),
                                          ('listening-ru-calls', (1236, 1237), '1236.wav')]:
                    texts = json.loads((ablate.ROOT/'examples/diagnostics'/f'{asset}.json').read_text())
                    expected = {(text, seed, ref) for group in texts.values() for text in group.values() for seed in seeds}
                    actual = {(c['text'], c['seed'], c['reference_audio']) for c in report['plan']['cases']
                              if c['filename'].startswith(asset+'/')}
                    self.assertEqual(actual, expected)
                wav = pack/'controls/timofey_seed1234.wav'
                self.assertEqual(sf.info(wav).subtype, 'FLOAT')
                self.assertGreater(sf.read(wav)[0].max(), 1.)
                ablate.main()
                self.assertEqual(gen.call_count, 58)
                wav.write_bytes(b'incomplete')
                ablate.main()
                self.assertEqual(gen.call_count, 59)
                self.assertEqual(ablate.file_digest(wav), report['results']['controls/timofey_seed1234.wav']['wav_sha256'])

    def test_dit_series_uses_original_cases_and_ungated_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)/'lora-base-recovery-4patch-0500'
            series = run/'audio/step-0500/series-2'
            originals = series/'lora'
            originals.mkdir(parents=True)
            ckpt = run/'lora/step-0500'
            ckpt.mkdir(parents=True)
            recipe = dict(mode='lora', base_sha256=ablate.BASE_SHA, acoustic_recovery=ablate.vox_recovery.FOUR_PATCH)
            (ckpt/'checkpoint.json').write_text(json.dumps(dict(step=500, recipe=recipe)))
            (ckpt/'adapter_config.json').write_text('{}')
            (ckpt/'adapter_model.safetensors').write_bytes(b'checkpoint')
            rows, prior = [], []
            for i, seed in enumerate((1236, 1239)):
                ref = run/f'ref-{i}.wav'
                sf.write(ref, np.zeros(1000), 24000)
                row = dict(id=f'cities_seed{seed}', seed=seed, text='Смотрите, по городам.', reference_audio=str(ref),
                           language='Russian', reference_text=f'Референс {i}.')
                rows.append(row)
                filename = f'case-{i:03d}.wav'
                (originals/filename).write_bytes(b'original')
                prior.append(dict(id=row['id'], text=row['text'], seed=seed, audio=f'lora/{filename}'))
            benchmark = series/'benchmark.jsonl'
            benchmark.write_text(''.join(json.dumps(row)+'\n' for row in rows))
            _, digest = ablate.load_benchmark(benchmark)
            (series/'report.json').write_text(json.dumps(dict(complete=True, step=500, settings=ablate.SETTINGS,
                benchmark_sha256=digest, recipes={'lora': recipe}, results={'lora': prior})))
            with (patch.object(ablate, 'model_identity', return_value=ablate.BASE_SHA),
                  patch.object(ablate, 'ModelManager') as manager, patch.object(ablate, 'route') as route,
                  patch.object(ablate, 'generate_case', return_value=(np.zeros(3000), 24000, {'finish_reason':'stop'})) as gen):
                route.return_value = {'backbone':0, 'dit':8}
                plan = ablate.series_plan(series)
                output = series/'dit'
                ablate.render(output, plan, audio_only=True)
                ablate.render(output, plan, audio_only=True)
                manager.return_value.load_base.assert_called_once_with(attention_backend='sdpa', adapter_path=str(ckpt))
                route.assert_called_once_with(manager.return_value.load_base.return_value.tts_core, 'dit')
                self.assertEqual([{k:c.args[1][k] for k in rows[0]} for c in gen.call_args_list], rows)
                self.assertTrue(all(c.kwargs['settings'] == ablate.SETTINGS for c in gen.call_args_list))
                self.assertEqual(sorted(p.name for p in output.glob('*.wav')), sorted(p.name for p in originals.iterdir()))
                self.assertTrue(all(p.read_bytes() == b'original' for p in originals.iterdir()))
                route.reset_mock()
                gen.reset_mock()
                with patch('sys.argv', ['ablation', '--series', str(originals), '--variants', 'backbone', 'both', '--audio-only']):
                    ablate.main()
                    ablate.main()
                self.assertEqual([call.args[1] for call in route.call_args_list], ['backbone', 'both'])
                self.assertEqual([{k:c.args[1][k] for k in rows[0]} for c in gen.call_args_list], rows*2)
                for variant in ('backbone', 'both'):
                    report = json.loads((series/variant/'report.json').read_text())
                    self.assertEqual(report['plan']['variants'], [variant])
                    self.assertEqual(set(report['results']), {p.name for p in originals.iterdir()})
                    self.assertTrue(all(row['group'] == f'step-0500/{variant}' for row in report['results'].values()))
                # The old single-DiT command still resumes the already rendered files.
                with patch('sys.argv', ['ablation', '--dit-series', str(originals), '--audio-only']):
                    ablate.main()
                self.assertEqual(gen.call_count, 4)
                backbone = series/'backbone'
                saved_report = (backbone/'report.json').read_bytes()
                route.reset_mock()
                gen.reset_mock()
                with patch('sys.argv', ['ablation', '--listen-from', str(backbone), '--audio-only']):
                    ablate.main()
                    ablate.main()
                route.assert_called_once_with(manager.return_value.load_base.return_value.tts_core, 'backbone')
                pack = backbone/'listening'
                self.assertEqual(len(list((pack/'short').glob('*.wav'))), 16)
                self.assertEqual(len(list((pack/'long').glob('*.wav'))), 8)
                plan = json.loads((pack/'report.json').read_text())['plan']
                self.assertEqual(plan['reference_case'], rows[0]['id'])
                self.assertEqual(plan['checkpoints']['step-0500']['path'], str(ckpt))
                self.assertEqual(len(gen.call_args_list), 24)
                for call, case in zip(gen.call_args_list, plan['cases'], strict=True):
                    generated = call.args[1]
                    self.assertEqual((generated['text'], generated['seed']), (case['text'], case['seed']))
                    self.assertIn(generated['seed'], (1236, 1237))
                    for key in ('reference_audio', 'reference_text', 'language'):
                        self.assertEqual(generated[key], rows[0][key])
                    self.assertFalse(call.kwargs['settings']['do_split'])
                self.assertEqual((backbone/'report.json').read_bytes(), saved_report)
                sf.write(rows[0]['reference_audio'], np.ones(1000), 24000)
                with self.assertRaisesRegex(ValueError, 'benchmark or reference'):
                    ablate.series_plan(series)

    def test_routes_only_the_requested_lora_deltas_and_restores_both(self):
        core = configure(tiny_core(), 'lora')
        layers = dict(core.get_base_model().named_modules())
        names = ['backbone_llm.layers.0.self_attn.q_proj', 'dit.blocks.0.attn.to_q']
        with torch.no_grad():
            for name in names:
                layers[name].lora_A['default'].weight.fill_(.1)
                layers[name].lora_B['default'].weight.fill_(.2)
            first = []
            for mode in ('both', 'backbone', 'dit', 'both'):
                counts = ablate.route(core, mode)
                for i, (name, group) in enumerate(zip(names, ('backbone', 'dit'))):
                    layer = layers[name]
                    x = torch.ones(1, layer.in_features)
                    out = layer(x)
                    enabled = mode in (group, 'both')
                    self.assertEqual(counts[group] > 0, enabled)
                    self.assertEqual(torch.equal(out, layer.base_layer(x)), not enabled)
                    if len(first) < 2:
                        first.append(out.clone())
                    if enabled:
                        torch.testing.assert_close(out, first[i], atol=0., rtol=0.)
                    self.assertFalse(any(p.requires_grad for p in layer.parameters()))

    def test_paired_checkpoints_shared_base_restart_and_short_audio(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp)
            output = parent/'matrix'
            ref = parent/'reference.wav'
            ref.write_bytes(b'original reference')
            fixtures = ((parent/'current', 100, ablate.vox_recovery.RESONANT),
                        (parent/'prev', 500, ablate.vox_recovery.FOUR_PATCH))
            for run, step, profile in fixtures:
                path = run/'lora'/f'step-{step:04d}'
                path.mkdir(parents=True)
                (path/'adapter_config.json').write_text('{}')
                (path/'adapter_model.safetensors').write_bytes(str(step).encode())
                gate = dict(percent=20, settings=ablate.SETTINGS, cases=[dict(id='cities_seed1237',
                    reference_audio=str(ref), reference_text='Да.', reference_sha256=ablate.file_digest(ref))])
                (path/'checkpoint.json').write_text(json.dumps(dict(step=step, validation=dict(accepted_updates=77),
                    recipe=dict(mode='lora', rise_gate=gate, dataset_sha256='data', base_sha256=ablate.BASE_SHA,
                                acoustic_recovery=profile))))
            generated = []
            def gen(tts, row, settings):
                self.assertFalse(settings['do_split'])
                self.assertEqual(row['reference_audio'], str(ref))
                generated.append((row['text'], row['seed']))
                return np.ones(3000, dtype=np.float32)*1.1, 24000, {'finish_reason':'stop'}
            with (patch.object(ablate, 'model_identity', return_value=ablate.BASE_SHA),
                  patch.object(ablate, 'ModelManager') as manager, patch.object(ablate, 'route', return_value={}) as route,
                  patch.object(ablate, 'generate_case', side_effect=gen)):
                ablate.run(output, parent/'current', parent/'prev', audio_only=True)
                self.assertEqual(manager.return_value.load_base.call_count, 3)
                saved = json.loads((output/'report.json').read_text())
                self.assertTrue(saved['audio_complete'])
                cases = [(c['text'], c['seed']) for c in saved['plan']['cases']]
                self.assertEqual(len(cases), 8)
                self.assertEqual(generated, cases*7)
                ablate.run(output, parent/'current', parent/'prev', audio_only=True)
                self.assertEqual(len(generated), 56)
                name = next(iter(saved['results']))
                (output/name).write_bytes(b'interrupted')
                ablate.run(output, parent/'current', parent/'prev', audio_only=True)
                self.assertEqual(len(generated), 57)
                with (output/'summary.csv').open() as stream:
                    rows = list(csv.DictReader(stream))
                self.assertEqual(len(rows), 56)
                self.assertTrue(all(row['contrast_15_19_db'] == '' for row in rows))
                with patch.object(ablate, 'load_asr') as load:
                    whisper = Mock()
                    load.return_value = (whisper, {})
                    whisper.transcribe.return_value = ([SimpleNamespace(text='Тест.')], None)
                    ablate.run(output, parent/'current', parent/'prev')
                    self.assertEqual(whisper.transcribe.call_count, 56)
                    scored = json.loads((output/'report.json').read_text())
                    self.assertTrue(scored['asr_complete'])
                    self.assertEqual(scored['results'][name]['transcript'], 'Тест.')
                    self.assertEqual(len(generated), 57)
                focused = parent/'dit-cities-timofey'
                route.reset_mock()
                manager.return_value.load_base.reset_mock()
                ablate.run(focused, parent/'current', parent/'prev', audio_only=True, dit_cities_timofey=True)
                report = json.loads((focused/'report.json').read_text())
                expected = [(f'{name}-seed{seed}', (ablate.ROOT/'examples/diagnostics'/f'long-{name}.txt').read_text().strip(), seed)
                            for name in ('cities', 'timofey') for seed in (1234, 1235)]
                self.assertEqual(generated[57:], [(text, seed) for _, text, seed in expected]*2)
                self.assertEqual({p.relative_to(focused).as_posix() for p in focused.rglob('*.wav')},
                                 {f'{stage}/dit/{case}.wav' for stage in ablate.STAGES for case, _, _ in expected})
                self.assertEqual([call.args[1] for call in route.call_args_list], ['dit', 'dit'])
                self.assertEqual([call.kwargs['adapter_path'] for call in manager.return_value.load_base.call_args_list],
                                 [report['plan']['checkpoints'][stage]['path'] for stage in ablate.STAGES])
                with (focused/'summary.csv').open() as stream:
                    self.assertTrue(all(row['tail_contrast_delta_vs_base_db'] == '' for row in csv.DictReader(stream)))
                ablate.run(focused, parent/'current', parent/'prev', audio_only=True, dit_cities_timofey=True)
                self.assertEqual(len(generated), 65)
                with self.assertRaisesRegex(ValueError, 'training recipe'):
                    conf = fixtures[0][0]/'lora/step-0100/checkpoint.json'
                    rec = json.loads(conf.read_text())
                    rec['recipe']['acoustic_recovery'] = ablate.vox_recovery.FOUR_PATCH
                    conf.write_text(json.dumps(rec))
                    ablate.run(output, parent/'current', parent/'prev', audio_only=True)

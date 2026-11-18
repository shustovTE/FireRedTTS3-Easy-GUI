"""Exercise the two independent FireRed updates on the real (downsized) module graph."""
import json
from contextlib import ExitStack
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf
import torch
from transformers import get_cosine_schedule_with_warmup

from test_training_batch import tiny_core
from fireredtts3_local.training_batch import set_training
from fireredtts3_local.vox_recipe_transfer import configure, objective, save_checkpoint, paired_data
from fireredtts3_local.vox_recipe_compare import content_flags
from fireredtts3_local import vox_recipe_transfer as transfer
from fireredtts3_local import voxcpm_data


def speaker_rows(root, speaker, lengths, corpus='dialogs'):
    rows = []
    for index, frames in enumerate(lengths):
        uid = f'{speaker}-{index}'
        path = str(root / f'{uid}.wav')
        sf.write(path, np.zeros(frames), 1000)
        rows.append(dict(utt_id=uid, dataset_id=corpus, speaker_id=speaker, audio=path,
                         text=f'Реплика {uid}' + ('?' if index % 2 == 0 else '.'),
                         duration=frames / 1000, source_audio=path, source_sha256=uid,
                         audio_sha256=uid, source_pcm_sha256=uid, pcm_sha256=uid))
    for index, row in enumerate(rows):
        ref = rows[(index + 1) % len(rows)]
        row.update(ref_utt_id=ref['utt_id'], ref_audio=ref['audio'], ref_sha256=ref['audio_sha256'],
                   ref_duration=ref['duration'], ref_pcm_sha256=ref['pcm_sha256'], ref_text=ref['text'])
    return rows


class TransferTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(1)

    def test_audited_draw_order_and_reference_text(self):
        with tempfile.TemporaryDirectory() as folder:
            d = Path(folder)
            report = dict(schema=voxcpm_data.FAIR_SCHEMA)
            (d/'report.json').write_text(json.dumps(report))
            # 9+7 seconds fit. The 5-second clip crosses the cap. The 3-second
            # clip would fit in the gap but falls after the first excluded clip.
            rows = speaker_rows(d, 'alice', [9000, 7000, 5000, 3000])
            held = speaker_rows(d, 'held', [20000, 18000])
            for name, data in [('validation-provenance.jsonl', held), ('train-provenance.jsonl', rows)]:
                (d/name).write_text(''.join(json.dumps(r)+'\n' for r in data))
            with patch.object(voxcpm_data, 'audit', return_value=report), patch.object(transfer, 'CAP_SECONDS', 20):
                a, unique, selection = paired_data(d)
                b, _, second = paired_data(d)
            self.assertEqual(a, b)
            self.assertEqual(selection, second)
            self.assertEqual(len(unique), 4)
            self.assertEqual(len(a['train']), 16000)
            self.assertEqual(sum('?' in t['text'] for _, t, _ in a['train']), 8000)
            self.assertEqual({r['utt_id'] for pair in a['train'] for r in pair[:2]}, {'alice-0', 'alice-1'})
            self.assertTrue(all(p['utt_id'] != t['utt_id'] and p['text'] == t['ref_text'] and
                                p['pcm_sha256'] == t['ref_pcm_sha256'] for p, t, _ in a['train']))
            self.assertEqual([(p['utt_id'], t['utt_id']) for p, t, _ in a['validation']],
                             [(r['ref_utt_id'], r['utt_id']) for r in held])
            self.assertEqual(selection['speakers']['alice'], dict(source_clips=4, source_seconds=24,
                             train_clips=2, train_seconds=16, excluded_by_cap=2, excluded_without_reference=0,
                             replaced_references=1))

    def test_cap_exact_sample_boundary_ties_and_independent_speakers(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(transfer, 'CAP_SECONDS', 20):
            root = Path(folder)
            exact = speaker_rows(root, 'a', [7300, 6350, 6350, 6350])
            over = speaker_rows(root, 'b', [7301, 6350, 6350, 3000])
            under = speaker_rows(root, 'c', [4000, 3000], 'spontaneous')
            selected, stats = transfer.cap_training_speakers(list(reversed(exact + over + under)))
            self.assertEqual([r['utt_id'] for r in selected], ['a-0', 'a-1', 'a-2', 'b-0', 'b-1', 'c-0', 'c-1'])
            self.assertEqual(stats['a']['train_seconds'], 20)
            self.assertEqual(stats['b']['train_seconds'], 13.651)
            self.assertEqual(stats['c']['excluded_by_cap'], 0)
            self.assertEqual(selected[-2:], under)

    def test_real_30_minute_cap_counts_decoded_samples(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            # At 1 kHz, the 61st 3-second clip would exceed the limit by one sample.
            exact = speaker_rows(root, 'exact', [30000]*59 + [27000, 3000])
            over = speaker_rows(root, 'over', [30000]*59 + [27001, 3000])
            _, stats = transfer.cap_training_speakers(exact + over)
            self.assertEqual(stats['exact']['train_seconds'], 1800)
            self.assertEqual(stats['exact']['train_clips'], 61)
            self.assertEqual(stats['over']['train_seconds'], 1797.001)
            self.assertEqual(stats['over']['excluded_by_cap'], 1)

    def test_does_not_reintroduce_short_reference_outside_cap(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(transfer, 'CAP_SECONDS', 20):
            rows = speaker_rows(Path(folder), 'a', [14000, 8000])
            selected, stats = transfer.cap_training_speakers(rows)
            self.assertEqual(selected, [])
            self.assertEqual(stats['a']['excluded_by_cap'], 1)
            self.assertEqual(stats['a']['excluded_without_reference'], 1)
            self.assertEqual(stats['a']['train_seconds'], 0)

    def test_modes_update_and_reload(self):
        for mode in ('lora','full'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                torch.manual_seed(5)
                base = tiny_core()
                original = {k: v.clone() for k,v in base.state_dict().items()}
                core = configure(base, mode)
                tts = SimpleNamespace(tts_core=core, device='cpu', _tokenize_text=lambda text: torch.tensor([1,2,3]))
                def payload(n):
                    z = torch.randn(1,n*2,4)
                    with torch.no_grad():
                        patch_emb = core.patch_encoder.eval()(z)
                    return dict(latents=z, speaker=torch.randn(1,6), patch_embeddings=patch_emb)
                row = dict(text='Привет.', language='Russian', utt_id='test')
                batch = [(payload(3), payload(n), row, row, 0.) for n in (3,6)]
                # The legacy cache must never enter the full graph.
                if mode == 'full':
                    for a,b,*_ in batch:
                        a['patch_embeddings'].fill_(float('nan'))
                        b['patch_embeddings'].fill_(float('nan'))
                set_training(core)
                params = [p for p in core.parameters() if p.requires_grad]
                opt = torch.optim.AdamW(params, lr=.001)
                schedule = get_cosine_schedule_with_warmup(opt, 0, 10)
                loss, metrics = objective(tts, batch, mode, 42, True)
                torch.testing.assert_close(loss.detach(), metrics['flow']+metrics['stop'])
                loss.backward()
                self.assertTrue(torch.isfinite(loss))
                self.assertEqual(mode == 'full', any(p.grad is not None for p in core.patch_encoder.parameters()))
                if mode == 'lora':
                    names = [n for n,p in core.named_parameters() if p.requires_grad]
                    self.assertTrue(all('lora_' in n and ('.self_attn.' in n or '.attn.' in n) for n in names))
                    self.assertTrue(any('.dit.' in n for n in names))
                opt.step()
                schedule.step()
                core.eval()
                expected, _ = objective(tts, batch, mode, 7, False)
                path = save_checkpoint(Path(tmp), core, opt, schedule, dict(mode=mode, base_sha256='toy'), 1, {})
                fresh = tiny_core()
                fresh.load_state_dict(original)
                tts.tts_core = configure(fresh, mode, path)
                tts.tts_core.eval()
                actual, _ = objective(tts, batch, mode, 7, False)
                torch.testing.assert_close(actual, expected)
                state = torch.load(path/'training_state.pt', map_location='cpu', weights_only=True)
                self.assertEqual(state['scheduler']['last_epoch'], 1)
                self.assertTrue(state['optimizer']['state'])

    def test_pilot_resume_reproduces_uninterrupted_training(self):
        rows = [dict(utt_id=str(i), audio_sha256=str(i), language="Russian", text=str(i)) for i in range(2)]
        pairs = [(rows[i], rows[1-i], 0.) for i in (0, 1)]
        class Codec(torch.nn.Module):
            sample_rate, downsample_rate = 1000, 16
            def __init__(self):
                super().__init__()
                self.decoder = torch.nn.Identity()
            def encode(self, wav, rate):
                return wav.reshape(1, -1, 4, 4).mean(-1)
        def manager():
            core = tiny_core().eval()
            tts = SimpleNamespace(tts_core=core, device='cpu', _tokenize_text=lambda s: torch.tensor([1, 2]),
                                  redae=Codec(), spk_extractor=torch.nn.Identity())
            return SimpleNamespace(load_base=lambda **kw: tts, unload=lambda: None)
        def cache(tts, *args):
            rng = torch.Generator().manual_seed(102)
            payloads = {}
            for row in rows:
                z = torch.randn((1, 8, 4), generator=rng)
                payloads[row['utt_id']] = dict(latents=z, speaker=torch.randn(1, 6, generator=rng),
                                               patch_embeddings=tts.tts_core.patch_encoder(z).detach())
            return SimpleNamespace(path=lambda r: Path(__file__), read_only=True, device=torch.device('cpu'),
                                   read=lambda r: payloads[r['utt_id']])
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as mocks:
            for row in rows:
                row['audio'] = str(Path(tmp) / f"{row['utt_id']}.wav")
                sf.write(row['audio'], np.sin(np.arange(125))*.2, 1000)
            for name, val in (('ModelManager', manager), ('FeatureCache', cache),
                              ('require_training_hardware', lambda: None), ('model_identity', lambda d: transfer.BASE_SHA),
                              ('paired_data', lambda d: ({'train': pairs*32, 'validation': pairs}, rows,
                                                        {'selection_sha256': 'toy'}))):
                mocks.enter_context(patch.object(transfer, name, val))
            from safetensors.torch import load_file
            for mode in ('lora', 'full'):
                root = Path(tmp)
                transfer.train(root, root/'ref', mode, 2, False)
                transfer.train(root, root/'pilot', mode, 1, False)
                transfer.train(root, root/'pilot', mode, 2, True)
                name = 'adapter_model.safetensors' if mode == 'lora' else 'fireredtts3_base/model.safetensors'
                def tensors(run):
                    return load_file(root/run/mode/'step-0002'/name)
                wanted, got = tensors('ref'), tensors('pilot')
                self.assertEqual(wanted.keys(), got.keys())
                for key in wanted:
                    torch.testing.assert_close(wanted[key], got[key], rtol=0, atol=0, msg=key)
                if mode == 'lora':
                    # The original already has a later checkpoint. Fork from the explicit pilot.
                    earlier = root/'pilot'/mode/'step-0001'
                    transfer.train(root, root/'fork', mode, 2, True, fork_from=earlier,
                                   shared_features=root/'pilot'/'features')
                    replayed = tensors('fork')
                    for key in wanted:
                        torch.testing.assert_close(wanted[key], replayed[key], rtol=0, atol=0, msg=key)
                    self.assertFalse((root/'fork'/mode/'step-0001').exists())
                    transfer.train(root, root/'recover', mode, 2, True, fork_from=earlier, recovery=True)
                    recovered = tensors('recover')
                    self.assertTrue(any(not torch.equal(got[k], recovered[k]) for k in got))
                    with self.assertRaisesRegex(ValueError, 'same recipe'):
                        transfer.train(root, root/'recover', mode, 3, True)
                    transfer.train(root, root/'recover', mode, 3, True, recovery=True)
                    transfer.train(root, root/'recover-direct', mode, 3, True, fork_from=earlier, recovery=True)
                    a, b = (load_file(root/r/mode/'step-0003'/name) for r in ('recover', 'recover-direct'))
                    for key in a:
                        torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
                    # Fresh recovery applies the new loss from the first microbatch.
                    with patch.object(transfer.vox_recovery, 'recovery_loss',
                                      wraps=transfer.vox_recovery.recovery_loss) as recovery_loss:
                        transfer.train(root, root/'fresh', mode, 1, False, recovery=True)
                        self.assertEqual(recovery_loss.call_count, transfer.ACCUM)
                    transfer.train(root, root/'fresh', mode, 2, True, recovery=True)
                    transfer.train(root, root/'fresh-direct', mode, 2, False, recovery=True)
                    info = json.loads((root/'fresh'/mode/'step-0002'/'checkpoint.json').read_text())
                    self.assertEqual(info['recipe']['recovery_start_step'], 0)
                    fresh, direct = tensors('fresh'), tensors('fresh-direct')
                    for key in fresh:
                        torch.testing.assert_close(fresh[key], direct[key], rtol=0, atol=0)

    def test_compare_uses_same_whole_cases_and_flags_bad_checkpoint(self):
        from fireredtts3_local import vox_recipe_compare as comp
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as mocks:
            root = Path(tmp)
            audio = root/'ref.wav'
            samples = np.ones(24000, dtype=np.float32)*.1
            sf.write(audio, samples, 24000, subtype='FLOAT')
            text = 'После обеда я зайду в ваш магазин.'
            bench = root/'input.jsonl'
            bench.write_text(json.dumps(dict(reference_audio=audio.name, reference_text='До встречи.',
                                             text=text, id='q', seed=7))+'\n')
            for mode in ('lora','full'):
                path = root/mode/'step-0030'
                path.mkdir(parents=True)
                record = dict(step=30, recipe=dict(base_sha256=transfer.BASE_SHA, dataset_sha256='audited', mode=mode))
                (path/'checkpoint.json').write_text(json.dumps(record))
            model = SimpleNamespace(tts_core=torch.nn.Identity())
            manager = SimpleNamespace(load_base=lambda **kw: model, unload=lambda: None)
            seen = []
            def generate(tts, case):
                seen.append((case['seed'],case['text'],case['reference_sha256']))
                return samples, 24000, dict(finish_reason='stop')
            def transcribe(path, **kw):
                hyp = text if Path(path).parent.name != 'full' else 'После обеда.'
                return [SimpleNamespace(text=hyp)], None
            scorer = SimpleNamespace(eval=lambda: lambda **kw: SimpleNamespace(embeddings=torch.ones(1,8)))
            selection = {'selection_sha256': 'audited'}
            for module, name, value in [
                (comp,'paired_data',lambda d: ({'train':[]}, [], selection)),
                (comp,'model_identity',lambda d: transfer.BASE_SHA), (comp,'generate_case',generate),
                (comp,'ModelManager',lambda:manager), (comp,'configure',lambda core, *args:core),
            ]:
                mocks.enter_context(patch.object(module,name,value))
            mocks.enter_context(patch('faster_whisper.WhisperModel', return_value=SimpleNamespace(transcribe=transcribe)))
            mocks.enter_context(patch('transformers.AutoFeatureExtractor.from_pretrained',return_value=lambda *a,**kw:{}))
            mocks.enter_context(patch('transformers.WavLMForXVector.from_pretrained',return_value=scorer))
            path = comp.compare(root,root,30,bench,'stub','stub')
            report = json.loads(path.read_text())
            self.assertEqual(len(seen),3)
            self.assertEqual(len(set(seen)),1)
            self.assertEqual(report['passed_automatic_checks'],dict(base=True,lora=True,full=False))
            self.assertEqual(sf.info(path.parent/'full'/'case-000.wav').subtype,'FLOAT')
            selection.update(original_selection_sha256='audited', selection_sha256='levelled')
            for mode in ('lora', 'full'):
                checkpoint = root/mode/'step-0030'/'checkpoint.json'
                updated = json.loads(checkpoint.read_text())
                updated['recipe']['dataset_sha256'] = 'levelled'
                checkpoint.write_text(json.dumps(updated))
            second = comp.compare(root, root, 30, None, 'stub', 'stub', against=path)
            result = json.loads(second.read_text())
            self.assertEqual(result['benchmark_sha256'], report['benchmark_sha256'])
            self.assertEqual(result['compared_to'], str(path))
            self.assertEqual(set(result['results']['full'][0]['delta_vs_unlevelled'].values()), {0.})
            # Continue LoRA on the same data. There is no Full checkpoint at step 500.
            later = root/'lora'/'step-0500'
            later.mkdir()
            record = json.loads((later.parent/'step-0030'/'checkpoint.json').read_text())
            record['step'] = 500
            (later/'checkpoint.json').write_text(json.dumps(record))
            seen.clear()
            third = comp.compare(root, root, 500, None, 'stub', 'stub', against=second, lora_only=True)
            resumed = json.loads(third.read_text())
            self.assertEqual(len(seen), 2)
            self.assertEqual(set(resumed['results']), {'base', 'lora'})
            self.assertEqual(resumed['benchmark_sha256'], result['benchmark_sha256'])
            self.assertEqual(resumed['reference_step'], 30)
            case = resumed['results']['lora'][0]
            self.assertEqual(set(case['delta_vs_previous_step'].values()), {0.})
            self.assertNotIn('delta_vs_unlevelled', case)
            with self.assertRaisesRegex(ValueError, 'recipe/selection mismatch'):
                comp.compare(root, root, 500, None, 'stub', 'stub', against=path, lora_only=True)
            record['recipe']['acoustic_recovery'] = transfer.vox_recovery.CONFIG
            (later/'checkpoint.json').write_text(json.dumps(record))
            ab = comp.compare(root, root, 500, None, 'stub', 'stub', against=third, lora_only=True)
            delta = json.loads(ab.read_text())['results']['lora'][0]['delta_vs_clean_control']
            self.assertEqual(set(delta.values()), {0.})
            seen.clear()
            # The old report has LoRA and Base with different metrics. Only Base should be used.
            cached = json.loads(second.read_text())
            cached['results']['base'][0]['final_voice_similarity'] = .75
            cached_path = second.with_name('cached-base.json')
            cached_path.write_text(json.dumps(cached))
            cached_eval = comp.compare(root, root, 500, None, 'stub', 'stub', against=cached_path,
                                       lora_only=True, reuse_base=True)
            reused = json.loads(cached_eval.read_text())
            self.assertEqual(len(seen), 1)
            self.assertFalse((cached_eval.parent/'base').exists())
            self.assertEqual((set(reused['results']), reused['reference_step']), ({'lora'}, 0))
            adapted = reused['results']['lora'][0]
            self.assertEqual(adapted['delta_vs_base']['final_voice_similarity'], adapted['final_voice_similarity'] - .75)
            self.assertEqual(reused['compared_to'], str(cached_path))
            cached['results']['base'][0]['seed'] += 1
            cached_path.write_text(json.dumps(cached))
            with self.assertRaisesRegex(ValueError, 'Invalid saved Base case'):
                comp.compare(root, root, 500, None, 'stub', 'stub', against=cached_path,
                             lora_only=True, reuse_base=True)
            self.assertEqual(len(seen), 1)
            del record['recipe']['acoustic_recovery']
            (later/'checkpoint.json').write_text(json.dumps(record))
            changed_ref = samples * .5
            sf.write(audio, changed_ref, 24000, subtype='FLOAT')
            with self.assertRaisesRegex(ValueError, 'differ from the original comparison'):
                comp.compare(root, root, 30, None, 'stub', 'stub', against=path)

    def test_fluent_omission_and_voice_drift_block_candidate(self):
        base = dict(finish_reason='stop', fraction_at_or_above_full_scale=0., end_minus_start_rms_db=0.,
                    voice_similarity=.8, final_voice_similarity=.8, asr={'missing_spans': [], 'wer': .01})
        row = dict(base, final_voice_similarity=.7, asr={'missing_spans':[{'words':5}], 'wer':.2})
        self.assertEqual(content_flags(base), [])
        self.assertIn('asr_missing_three_consecutive_words', content_flags(row, base))
        self.assertIn('final_voice_similarity_drop', content_flags(row, base))

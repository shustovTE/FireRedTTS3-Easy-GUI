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
from fireredtts3_local import voxcpm_data, voxcpm_balance


class TransferTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(1)

    def test_audited_draw_order_and_reference_text(self):
        with tempfile.TemporaryDirectory() as folder:
            d = Path(folder)
            report = dict(schema=voxcpm_data.FAIR_SCHEMA)
            (d/'report.json').write_text(json.dumps(report))
            rows = [dict(audio=f'{i}.wav', text=f'hello {i}', ref_text=f'hello {1-i}',
                         ref_audio=f'{1-i}.wav', audio_sha256=str(i)) for i in range(2)]
            for name, data in [('validation-provenance.jsonl', rows), ('train-provenance.jsonl', rows),
                               (voxcpm_balance.MANIFEST, rows*8000)]:
                (d/name).write_text(''.join(json.dumps(r)+'\n' for r in data))
            with patch.object(voxcpm_data, 'audit', return_value=report):
                a, unique, _ = paired_data(d)
                b, _, _ = paired_data(d)
            self.assertEqual(a, b)
            self.assertEqual(len(unique), 2)
            self.assertEqual(len(a['train']), 16000)
            self.assertEqual(sum(t['text'] == 'hello 0' for _,t,_ in a['train']), 8000)
            self.assertTrue(all(p['text'] == t['ref_text'] for p,t,_ in a['train']))

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
        def manager():
            core = tiny_core().eval()
            tts = SimpleNamespace(tts_core=core, device='cpu', _tokenize_text=lambda s: torch.tensor([1, 2]),
                                  redae=torch.nn.Identity(), spk_extractor=torch.nn.Identity())
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
            for name, val in (('ModelManager', manager), ('FeatureCache', cache),
                              ('require_training_hardware', lambda: None), ('model_identity', lambda d: transfer.BASE_SHA),
                              ('paired_data', lambda d: ({'train': pairs*16, 'validation': pairs}, rows, 'toy'))):
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
            for module, name, value in [
                (comp,'paired_data',lambda d: ({'train':[]}, [], 'audited')),
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

    def test_fluent_omission_and_voice_drift_block_candidate(self):
        base = dict(finish_reason='stop', fraction_at_or_above_full_scale=0., end_minus_start_rms_db=0.,
                    voice_similarity=.8, final_voice_similarity=.8, asr={'missing_spans': [], 'wer': .01})
        row = dict(base, final_voice_similarity=.7, asr={'missing_spans':[{'words':5}], 'wer':.2})
        self.assertEqual(content_flags(base), [])
        self.assertIn('asr_missing_three_consecutive_words', content_flags(row, base))
        self.assertIn('final_voice_similarity_drop', content_flags(row, base))

import json
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf
import torch
from safetensors.torch import load_file

from fireredtts3_local import redae_decoder_training as decoder
from fireredtts3_local import train_backbone_attention_250 as job, vox_recipe_transfer as transfer
from fireredtts3_local.history_probe import read_audio
from fireredtts3_local.model_manager import ModelManager
from fireredtts3_local.training_data import file_digest
from test_training_batch import tiny_core
from test_train_backbone_attention_250 import prepare_pilots


def tiny_tts():
    ModelManager._ensure_upstream_namespace()
    from fireredtts3.redae.redae import RedAE, RedAEConfig
    config = dict(audio_patch_size=32, audio_sample_rate=24000, bottleneck_dim=4, enc_downsample_num_hidden_layers=1)
    for side in ('enc', 'dec'):
        for key, val in dict(hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                             num_attention_heads=2, num_key_value_heads=1).items():
            config[side+'_'+key] = val
    with torch.random.fork_rng(devices=[]), patch.dict(os.environ, FIRERED_ATTN_IMPLEMENTATION='sdpa'):
        torch.manual_seed(119)
        return SimpleNamespace(device='cpu', redae=RedAE(RedAEConfig(**config)).eval(), tts_core=tiny_core().eval(),
                               spk_extractor=torch.nn.Identity(), _tokenize_text=lambda _: torch.tensor([1, 2]))


class DecoderTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(1)

    def test_two_real_adapters_update_resume_and_render(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = []
            for i in range(2):
                p = root/f'{i}.wav'
                sf.write(p, np.sin(np.arange(5301+i)*.083)*.1, 16000)
                rows.append(dict(utt_id=str(i), audio=str(p), audio_sha256=file_digest(p), text='Да.', language='Russian'))
            tts = tiny_tts()
            features = {}
            with torch.no_grad():
                for row in rows:
                    _, padded, left = read_audio(tts, row)
                    self.assertGreater(left, 0)
                    z = tts.redae.encode(padded, 24000).float()
                    features[row['utt_id']] = dict(latents=z, patch_embeddings=tts.tts_core.patch_encoder(z),
                                                  speaker=torch.ones(1, 6))
            pairs = [(rows[0], rows[1], 0.), (rows[1], rows[0], 0.)]
            cache = SimpleNamespace(path=lambda _: Path(__file__), device=torch.device('cpu'),
                                    read_only=True, read=lambda r: features[r['utt_id']])
            flags = dict(redae_decoder=True, backbone_only=True, recovery=transfer.vox_recovery.FOUR_PATCH)
            with (patch.object(transfer, 'require_training_hardware'), patch.object(transfer, 'ACCUM', 1),
                  patch.object(transfer, 'FeatureCache', return_value=cache),
                  patch.object(transfer, 'model_identity', return_value=transfer.BASE_SHA),
                  patch.object(transfer, 'ModelManager', side_effect=lambda: SimpleNamespace(load_base=lambda **_: tiny_tts(),
                                                                                              unload=lambda: None)),
                  patch.object(transfer, 'paired_data', return_value=({'train': pairs*3, 'validation': pairs},
                                                                      rows, {'selection_sha256': 'test'}))):
                transfer.train(root, root/'direct', 'lora', 3, False, **flags)
                transfer.train(root, root/'resume', 'lora', 2, False, **flags)
                transfer.train(root, root/'resume', 'lora', 3, True, **flags)
                transfer.train(root, root/'backbone', 'lora', 3, False, **dict(flags, redae_decoder=False))

            path = root/'direct/lora/step-0003'
            for part in (Path('adapter_model.safetensors'), Path(decoder.FOLDER)/'adapter_model.safetensors'):
                direct = load_file(path/part)
                resumed = load_file(root/'resume/lora/step-0003'/part)
                self.assertTrue(any('lora_B' in k and v.count_nonzero() for k, v in direct.items()))
                if len(part.parts) == 1:
                    self.assertTrue(all('.backbone_llm.' in k for k in direct))
                    torch.testing.assert_close(direct, load_file(root/'backbone/lora/step-0003'/part), atol=0, rtol=0)
                torch.testing.assert_close(direct, resumed, atol=0, rtol=0)
            metrics = json.loads((path/'checkpoint.json').read_text())['validation']
            self.assertGreater(metrics['decoder_clean'], 0)
            self.assertGreater(metrics['decoder_damaged'], 0)
            clean_model = tiny_tts()
            z = features['0']['latents']
            orig = clean_model.redae.decode(z)[0]
            self.assertTrue(decoder.has_decoder(path))
            decoder.configure(clean_model.redae, path, training=False)
            changed = clean_model.redae.decode(z)[0]
            self.assertFalse(torch.equal(orig, changed))
            self.assertFalse(any(p.requires_grad for p in clean_model.redae.parameters()))
            self.assertTrue(torch.isfinite(changed).all())
            with (path/decoder.FOLDER/'adapter_model.safetensors').open('ab') as f:
                f.write(b'x')
            with self.assertRaisesRegex(ValueError, 'incomplete RedAE decoder'):
                decoder.configure(tiny_tts().redae, path, training=False)

    def test_listening_loads_both_adapters_only_at_requested_steps_and_isolates_base(self):
        original = tiny_tts()
        codec_type = type(original.redae)
        def make_base(*args, **kwargs):
            tts = tiny_tts()
            tts.redae = codec_type.from_pretrained('fixture')
            return tts

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            prepare_pilots(root)
            run = root/job.DECODER_RUN
            trained_steps = []
            def save(data, out, mode, until, **kwargs):
                self.assertTrue(kwargs['redae_decoder'] and kwargs['backbone_only'])
                self.assertEqual(kwargs['resume'], until == 500)
                self.assertIsNone(kwargs['fork_from'])
                trained_steps.append(until)
                tts = tiny_tts()
                core = transfer.configure(tts.tts_core, 'lora', backbone_only=True)
                dec = decoder.configure(tts.redae, training=True)
                with torch.no_grad():
                    for n, p in dec.named_parameters():
                        if 'lora_B' in n:
                            p.fill_(until / 10000)
                dest = out/'lora'
                dest.mkdir(exist_ok=True)
                opt = torch.optim.AdamW(core.parameters())
                recipe = json.loads((out/'replay.json').read_text())['recipe']
                transfer.save_checkpoint(dest, core, opt, torch.optim.lr_scheduler.StepLR(opt, 1),
                                         recipe, until, {}, decoder_adapter=dec)

            def generate(tts, row, *, settings):
                probs = [p for n, p in tts.redae.decoder.qwen3.named_parameters() if 'lora_B' in n]
                self.assertTrue(probs)
                self.assertAlmostEqual(probs[0].flatten()[0].item(), trained_steps[-1]/10000)
                return np.ones(2400, dtype=np.float32)*.1, 24000, {'finish_reason': 'stop'}

            with (patch.object(transfer, 'train', side_effect=save), patch.object(job.listening, 'recognize'),
                  patch.object(job.listening, 'generate_case', side_effect=generate) as generator,
                  patch.object(ModelManager, 'ensure_weights'),
                  patch.dict(sys.modules, {'fireredtts3.core': SimpleNamespace(FireRedTTS3=make_base)}),
                  patch.object(codec_type, 'from_pretrained'),
                  patch.object(codec_type, '_easy_gui_original_from_pretrained',
                               staticmethod(lambda *_: tiny_tts().redae), create=True)):
                job.run(root, root/'data', backbone_redae_to_500=True)
                job.run(root, root/'data', backbone_redae_to_500=True)
                self.assertEqual(trained_steps, [250, 500])
                self.assertEqual(generator.call_count, 116)
                for step in ('step-0250', 'step-0500'):
                    folder = run/'audio'/step/'backbone-redae-complete'
                    report = json.loads((folder/'report.json').read_text())
                    self.assertTrue(report['audio_complete'])
                    self.assertEqual(len(report['results']), 58)
                    self.assertTrue(all(r['active_layers'] == dict(backbone=8, dit=0, redae_decoder=4)
                                        for r in report['results'].values()))
                    self.assertIn('redae_decoder/adapter_model.safetensors', report['plan']['checkpoints'][step]['hashes'])
                manager = ModelManager()
                baseline = manager.load_base(attention_backend='sdpa').redae
                adapted = manager.load_base(attention_backend='sdpa', adapter_path=run/'lora/step-0500').redae
                self.assertIsNot(baseline, adapted)
                self.assertIs(manager.load_base(attention_backend='sdpa').redae, baseline)
                self.assertFalse(any('lora_' in n for n, p in baseline.named_parameters()))

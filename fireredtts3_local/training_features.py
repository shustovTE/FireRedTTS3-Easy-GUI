"""Versioned frozen features and bounded, CPU-only prefetch for training."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
import pickle
from pathlib import Path
import time
from types import SimpleNamespace
import uuid

import torch
import torchaudio

from .training_data import file_digest, model_identity


class FeatureCache:
    VERSION = 'redae-campp-patches-v2'

    def __init__(self, tts, root=None, model_fingerprint='', read_only=False):
        self.tts = tts
        self.read_only = read_only
        self.device = torch.device(tts.device)
        self.spec = {'version': self.VERSION, 'model': model_fingerprint,
                     'precision': 'cuda-bfloat16' if self.device.type == 'cuda' else 'cpu-float32',
                     'torch': str(torch.__version__),
                     'attention': os.environ.get('FIRERED_ATTN_IMPLEMENTATION', 'sdpa')}
        signature = hashlib.sha256(json.dumps(self.spec, sort_keys=True).encode()).hexdigest()
        self.root = Path(root) / self.VERSION / signature if root else None
        if self.root and not model_fingerprint:
            raise ValueError('Disk features require a model fingerprint')
        if read_only and self.root is None:
            raise ValueError('Precomputed mode requires cache_dir')
        if self.root and not read_only:
            self.root.mkdir(parents=True, exist_ok=True)

    def path(self, row):
        if not self.root:
            raise ValueError('No feature cache directory')
        key = row['audio_sha256']
        return self.root / key[:2] / f'{key}.pt'

    def validate(self, payload, row):
        if not isinstance(payload, dict) or payload.get('spec') != self.spec or payload.get('audio_sha256') != row['audio_sha256']:
            raise ValueError('Feature metadata/hash mismatch; rerun precompute --repair')
        latents, speaker, patches = (payload.get(k) for k in ('latents', 'speaker', 'patch_embeddings'))
        if not all(isinstance(t, torch.Tensor) and t.dtype == torch.float32 and t.device.type == 'cpu' for t in (latents, speaker, patches)):
            raise ValueError('Features must contain float32 CPU tensors')
        patch_size = self.tts.tts_core.patch_size
        if latents.ndim != 3 or latents.shape[0] != 1 or latents.shape[1] < patch_size or latents.shape[1] % patch_size:
            raise ValueError('Invalid cached latent sequence')
        if speaker.ndim != 2 or speaker.shape[0] != 1:
            raise ValueError('Invalid cached speaker embedding')
        if patches.ndim != 3 or patches.shape[:2] != (1, latents.shape[1] // patch_size):
            raise ValueError('Invalid cached patch sequence')
        encoder = self.tts.tts_core.patch_encoder
        if latents.shape[-1] != getattr(encoder, 'in_dim', latents.shape[-1]) or patches.shape[-1] != getattr(encoder, 'out_dim', patches.shape[-1]):
            raise ValueError('Cached feature dimensions do not match PatchEncoder')
        config = getattr(self.tts.tts_core, 'config', None)
        if speaker.shape[-1] != getattr(config, 'spk_in_dim', speaker.shape[-1]):
            raise ValueError('Cached speaker dimension does not match the core')
        if not all(torch.isfinite(t).all() for t in (latents, speaker, patches)):
            raise ValueError('Non-finite cached features')
        return {'latents': latents, 'speaker': speaker, 'patch_embeddings': patches}

    def read(self, row):
        path = self.path(row)
        if not path.is_file():
            raise FileNotFoundError(f'Missing precomputed features for {row.get("utt_id", row["audio"])}. Run adapt precompute with this recipe first.')
        try:
            return self.validate(torch.load(path, map_location='cpu', weights_only=True), row)
        except (ValueError, RuntimeError, EOFError, pickle.UnpicklingError, IndexError, KeyError, TypeError) as exc:
            raise ValueError(f'Invalid feature cache {path}; use precompute --repair') from exc

    def require_all(self, rows):
        missing = [row['utt_id'] for row in rows if not self.path(row).is_file()]
        if missing:
            raise ValueError(f'{len(missing)} missing feature files; first {missing[:3]}. Run adapt precompute with this recipe before training.')

    def encode(self, row):
        from .training_objective import autocast
        if self.read_only:
            raise RuntimeError('Audio encoding is forbidden in precomputed training mode')
        if file_digest(row['audio']) != row['audio_sha256']:
            raise ValueError('Audio changed since manifest validation')
        audio, sr = torchaudio.load(row['audio'])
        if audio.shape[0] != 1 or not torch.isfinite(audio).all():
            raise ValueError(f'Expected finite mono audio: {row["audio"]}')
        encoder = self.tts.tts_core.patch_encoder
        if any(isinstance(m, torch.nn.modules.dropout._DropoutNd) and m.training for m in encoder.modules()):
            # Both supported adaptation profiles keep PatchEncoder frozen.
            encoder.eval()
        with torch.no_grad():
            audio = torchaudio.functional.resample(audio, sr, self.tts.redae.sample_rate)
            audio = self.tts.redae.pad_to_multiple_of(audio, self.tts.redae.downsample_rate*self.tts.tts_core.patch_size).to(self.device)
            latents = self.tts.redae.encode(audio, self.tts.redae.sample_rate).float()
            speaker = self.tts.spk_extractor(audio, self.tts.redae.sample_rate).float()
            with autocast(self.device):
                patches = encoder(latents)
        payload = {'spec': self.spec, 'audio_sha256': row['audio_sha256'],
                   'latents': latents.float().cpu().clone(), 'speaker': speaker.float().cpu().clone(),
                   'patch_embeddings': patches.float().cpu().clone()}
        self.validate(payload, row)
        if self.root:
            path = self.path(row)
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(f'.{uuid.uuid4().hex}.tmp')
            try:
                torch.save(payload, temporary)
                os.replace(temporary, path)
            finally:
                temporary.unlink(missing_ok=True)
        return self.validate(payload, row)

    def get_features(self, row):
        if self.root and self.path(row).exists():
            return self.read(row)
        if self.read_only:
            return self.read(row)  # actionable missing-cache error
        return self.encode(row)

    def get(self, row):
        # Backwards-compatible access for scalar validation/reference tests.
        data = self.get_features(row)
        return data['latents'].to(self.device), data['speaker'].to(self.device)


def precompute_features(tts, rows, cache_dir, fingerprint, repair=False, log=print):
    cache = FeatureCache(tts, cache_dir, fingerprint)
    unique = {row['audio_sha256']: row for row in rows}
    report = {'spec': cache.spec, 'unique_audio': len(unique), 'encoded': 0, 'reused': 0, 'complete': False}
    started = time.perf_counter()
    report_path = cache.root / 'precompute_report.json'
    for index, row in enumerate(unique.values(), 1):
        reused = False
        if cache.path(row).exists():
            try:
                cache.read(row)
                report['reused'] += 1
                reused = True
            except ValueError:
                if not repair:
                    raise
        if not reused:
            cache.encode(row)
            report['encoded'] += 1
        if index % 100 == 0 or index == len(unique):
            log(f'[precompute] {index}/{len(unique)} clips; elapsed={time.perf_counter()-started:.1f}s')
    report.update(complete=True, elapsed_seconds=time.perf_counter()-started)
    temporary = report_path.with_suffix(f'.{uuid.uuid4().hex}.tmp')
    temporary.write_text(json.dumps(report, indent=2))
    temporary.replace(report_path)
    log(f'[precompute] Complete: {report}')
    return report


def load_feature_pipeline(base, device='cuda', attention_backend='sdpa'):
    """Load only RedAE, CAM++ and PatchEncoder; no Qwen backbone or DiT."""
    from .model_manager import ModelManager
    ModelManager._ensure_upstream_namespace()
    from fireredtts3.redae.redae import RedAE
    from fireredtts3.campp.campp import CamppEmbedding
    from fireredtts3.llm.patch_encoder import PatchEncoder
    from fireredtts3.llm.fireredtts3_base import FireRedTTS3BaseCoreConfig
    from safetensors import safe_open
    base = Path(base)
    config = FireRedTTS3BaseCoreConfig.from_pretrained(base/'fireredtts3_base')
    state = {}
    for shard in sorted((base/'fireredtts3_base').glob('*.safetensors')):
        with safe_open(shard, framework='pt', device='cpu') as handle:
            for key in handle.keys():
                if key.startswith('patch_encoder.'):
                    state[key.removeprefix('patch_encoder.')] = handle.get_tensor(key)
    if 'out_proj.linear.weight' not in state:
        raise ValueError('Cannot find original PatchEncoder tensors')
    patch_encoder = PatchEncoder(in_dim=config.redae_dim, out_dim=state['out_proj.linear.weight'].shape[0],
                                 patch_size=config.patch_size, hidden_size=config.patch_encoder_hidden_size,
                                 mlp_ratio=config.patch_encoder_mlp_ratio, depth=config.patch_encoder_depth,
                                 num_heads=config.patch_encoder_num_heads)
    patch_encoder.load_state_dict(state, strict=True)
    patch_encoder.rotary_embed.rope_init()
    previous = os.environ.get('FIRERED_ATTN_IMPLEMENTATION')
    try:
        os.environ['FIRERED_ATTN_IMPLEMENTATION'] = attention_backend
        # Bypass the GUI's shared-instance hook when precomputing another base.
        load_redae = getattr(RedAE, '_easy_gui_original_from_pretrained', RedAE.from_pretrained)
        redae = load_redae(base/'redae').to(device).eval()
    finally:
        if previous is None:
            os.environ.pop('FIRERED_ATTN_IMPLEMENTATION', None)
        else:
            os.environ['FIRERED_ATTN_IMPLEMENTATION'] = previous
    speaker = CamppEmbedding(str(base/'campp'/'campplus_voxceleb.bin')).to(device).eval()
    patch_encoder = patch_encoder.to(device).eval()
    for module in (redae, speaker, patch_encoder):
        module.requires_grad_(False)
    return SimpleNamespace(device=torch.device(device), redae=redae, spk_extractor=speaker,
                           tts_core=SimpleNamespace(patch_encoder=patch_encoder, patch_size=config.patch_size, config=config))


def precompute_recipe(recipe_path, base, repair=False, log=print):
    from .training_run import TrainingOptions, prepare_run_data, require_training_hardware
    recipe = json.loads(Path(recipe_path).read_text(encoding='utf-8'))
    options = TrainingOptions.parse(recipe.get('options'))
    if not options.cache_dir:
        raise ValueError('Recipe needs cache_dir')
    train, validation, replay, adaptation = prepare_run_data(recipe['manifest_path'], options, recipe.get('seed', 1234), include_adaptation_validation=True)
    rows = train.rows + validation.rows + (replay.rows if replay else []) + (adaptation.rows if adaptation else [])
    require_training_hardware()
    fingerprint = model_identity(base)
    pipeline = load_feature_pipeline(base, attention_backend=options.attention_backend)
    previous = os.environ.get('FIRERED_ATTN_IMPLEMENTATION')
    try:
        os.environ['FIRERED_ATTN_IMPLEMENTATION'] = options.attention_backend
        return precompute_features(pipeline, rows, options.cache_dir, fingerprint, repair, log)
    finally:
        if previous is None:
            os.environ.pop('FIRERED_ATTN_IMPLEMENTATION', None)
        else:
            os.environ['FIRERED_ATTN_IMPLEMENTATION'] = previous
        del pipeline
        torch.cuda.empty_cache()


@dataclass
class BatchRequest:
    pairs: list
    sampler_state: tuple | None = None


class BatchFeatures:
    """Bounded CPU prefetch. No CUDA tensors/models run in loader threads."""
    def __init__(self, cache, workers=2, prefetch=2):
        if workers < 0 or prefetch < 1:
            raise ValueError("Invalid loader workers/prefetch")
        self.cache = cache
        self.workers, self.prefetch = workers, prefetch
        self.pool = ThreadPoolExecutor(max_workers=workers) if cache.read_only and workers else None

    def close(self):
        if self.pool:
            self.pool.shutdown(wait=True, cancel_futures=True)

    def _load(self, request):
        pairs = request.pairs
        unique = {}
        for prompt, target, _ in pairs:
            for row in (prompt, target):
                if row['audio_sha256'] not in unique:
                    data = self.cache.read(row) if self.cache.read_only else self.cache.get_features(row)
                    if self.cache.read_only and self.cache.device.type == 'cuda':
                        data = {key: value.pin_memory() for key, value in data.items()}
                    unique[row['audio_sha256']] = data
        return request, [(unique[p['audio_sha256']], unique[t['audio_sha256']], p, t, keep) for p, t, keep in pairs]

    def iterate(self, batches):
        if self.pool is None:
            for pairs in batches:
                yield self._load(pairs)
            return
        pending = deque()
        source = iter(batches)
        for _ in range(self.prefetch):
            pairs = next(source, None)
            if pairs is not None:
                pending.append(self.pool.submit(self._load, pairs))
        while pending:
            result = pending.popleft().result()
            pairs = next(source, None)
            if pairs is not None:
                pending.append(self.pool.submit(self._load, pairs))
            yield result

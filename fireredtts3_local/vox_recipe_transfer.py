"""Compare the pinned VoxCPM2 LoRA/full SFT presets on native FireRedTTS3 Base."""
from __future__ import annotations

import argparse
from collections import defaultdict
from fractions import Fraction
import json
from pathlib import Path
import random
import tempfile

from filelock import FileLock
import soundfile as sf
import torch

from . import vox_recipe_levels, voxcpm_balance, voxcpm_data
from . import vox_recovery
from .model_manager import ModelManager
from .paths import MODEL_DIR
from .training_batch import batch_flow_loss, set_training
from .training_data import file_digest, model_identity, plain_text
from .training_features import BatchFeatures, BatchRequest, FeatureCache, precompute_features
from .training_job import write_json
from .training_objective import adapter_targets
from .training_run import isolated_rng, require_training_hardware, to_examples
from .workspace_paths import local_path

OUTPUT = 'training/outputs/firered-vox-recipe-12-cap30m'
CAP_SECONDS = 1800
BASE_SHA = 'd41f62029f89c6837d5f5b16248aceb695ea362136c54c2ea8011f4e41390f25'
UPSTREAM = '300e4260b509066151d623dc9c0aa922cf82232b'
SEED = 1234
TOTAL, MICRO, ACCUM = 1000, 2, 8
LR = {'lora': 1e-4, 'full': 1e-5}


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def cap_training_speakers(rows):
    """Keep a longest-first prefix per speaker, measured in decoded WAV samples."""
    speakers = defaultdict(list)
    for row in rows:
        info = sf.info(local_path(row['audio']))
        duration = Fraction(info.frames, info.samplerate)
        speakers[row['speaker_id']].append((duration, row))

    selected, stats = [], {}
    for speaker, clips in sorted(speakers.items()):
        clips.sort(key=lambda clip: (-clip[0], clip[1]['utt_id']))
        used = Fraction()
        retained = []
        for duration, row in clips:
            if used + duration > CAP_SECONDS:
                break
            retained.append((duration, row))
            used += duration

        # Source speaker IDs include the corpus namespace. Keep both sides of
        # every training pair inside the same capped pool.
        pool = [row for _, row in retained]
        available = {row['audio'] for row in pool}
        missing = replaced = 0
        for duration, row in retained:
            if row['ref_audio'] in available:
                selected.append(row)
                continue
            ref = voxcpm_data._reference(row, pool)
            if ref is None:
                missing += 1
                used -= duration
                continue
            selected.append(dict(row, ref_audio=ref['audio'], ref_text=ref['text'],
                                 ref_utt_id=ref['utt_id'], ref_duration=ref['duration'],
                                 ref_sha256=ref['audio_sha256'], ref_pcm_sha256=ref['pcm_sha256']))
            replaced += 1
        stats[speaker] = dict(source_clips=len(clips), source_seconds=float(sum(d for d, _ in clips)),
                              train_clips=len(pool) - missing, train_seconds=float(used),
                              excluded_by_cap=len(clips) - len(pool), excluded_without_reference=missing,
                              replaced_references=replaced)
    return selected, stats


def paired_data(dataset):
    source, levels = dataset, None
    if (dataset / vox_recipe_levels.REPORT).is_file():
        source, levels = vox_recipe_levels.read_report(dataset)
    report = voxcpm_data.audit(source)
    if report['schema'] != voxcpm_data.FAIR_SCHEMA:
        raise ValueError('Expected the confirmed Dialogs + Spontaneous 1000-update preparation')
    original = read_rows(source / 'train-provenance.jsonl')
    train, speakers = cap_training_speakers(original)
    missing_corpora = {r['dataset_id'] for r in original} - {r['dataset_id'] for r in train}
    if missing_corpora:
        raise ValueError(f'No same-speaker pairs remain after the duration cap in {sorted(missing_corpora)}')
    repeats, balance = voxcpm_balance.even_repeats(train, [r['dataset_id'] for r in train],
                                                  draws=TOTAL * MICRO * ACCUM)
    validation = read_rows(source / 'validation-provenance.jsonl')
    source_sha = file_digest(source / 'report.json')
    identity = dict(source_sha256=source_sha, policy='longest-prefix-v1', cap_seconds=CAP_SECONDS,
                    seed=SEED, train=[(r['audio_sha256'], r['ref_sha256'], n)
                                     for r, n in zip(train, repeats, strict=True)])
    fingerprint = voxcpm_data._digest_json(identity)
    selection = dict(source_report_sha256=source_sha, selection_sha256=fingerprint,
                     cap_seconds=CAP_SECONDS, speakers=speakers, balance=balance,
                     train_hours=sum(s['train_seconds'] for s in speakers.values()) / 3600,
                     validation_clips=len(validation))
    if levels is not None:
        vox_recipe_levels.verify_files(dataset, levels, {r['audio_sha256'] for r in original + validation})
        selection['original_selection_sha256'] = fingerprint
        selection['level_report_sha256'] = file_digest(dataset / vox_recipe_levels.REPORT)
        selection['selection_sha256'] = voxcpm_data._digest_json([fingerprint, selection['level_report_sha256']])
        selection['level_summary'] = {split: vox_recipe_levels.summarize(levels['files'], rows) for split, rows
                                      in [('train', train), ('validation', validation)]}
    splits, unique = {}, {}
    for split, rows, counts in [('train', train, repeats), ('validation', validation, [1]*len(validation))]:
        native = {r['audio']: dict(r, audio=str(local_path(r['audio'])), text=plain_text(r['text']),
                                   language='Russian') for r in rows}
        if levels is not None:
            for row in native.values():
                for field, sha_field, pcm_field in [('audio', 'audio_sha256', 'pcm_sha256'),
                                                   ('ref_audio', 'ref_sha256', 'ref_pcm_sha256')]:
                    sha = row[sha_field]
                    record = levels['files'][sha]
                    row['unlevelled_' + field] = str(local_path(row[field]))
                    row['unlevelled_' + sha_field] = sha
                    row[field] = str((dataset / 'audio' / f'{sha}.wav').resolve())
                    row[sha_field], row[pcm_field] = record['audio_sha256'], record['pcm_sha256']
        unique.update((r['audio_sha256'], r) for r in native.values())
        pairs = []
        for target, n in zip(rows, counts, strict=True):
            pairs.extend([(native[target['ref_audio']], native[target['audio']], 0.)] * n)
        splits[split] = pairs
    random.Random(SEED).shuffle(splits['train'])
    return splits, list(unique.values()), selection


def configure(core, mode, checkpoint=None):
    if mode == 'full':
        if checkpoint:
            from safetensors.torch import load_model
            load_model(core, checkpoint / 'fireredtts3_base' / 'model.safetensors', strict=True)
        return core.float().requires_grad_(True)
    from peft import LoraConfig
    from .hybrid import create_adapter, load_adapter
    if checkpoint:
        return load_adapter(core, checkpoint, is_trainable=True)
    return create_adapter(core, LoraConfig(r=32, lora_alpha=32, lora_dropout=0.,
                                          target_modules=adapter_targets(core, 'vox_attention'), bias='none'))


def batches(pairs):
    for i in range(0, len(pairs), MICRO):
        yield BatchRequest(pairs[i:i+MICRO])


def objective(tts, loaded, mode, seed, train):
    examples = to_examples(tts, loaded, 'native', use_cached_patches=mode == 'lora')
    generator = torch.Generator(device=tts.device).manual_seed(seed)
    return batch_flow_loss(tts, examples, patches_per_example='all', cfg_dropout=.1 if train else 0.,
                           stop_weight=1., generator=generator, dit_chunk_size=64,
                           precision_policy='native', vox_recipe=True)


@torch.no_grad()
def validate(tts, loader, pairs, mode):
    set_training(tts.tts_core, False)
    totals = {}
    with isolated_rng(SEED):
        for i, (_, loaded) in enumerate(loader.iterate(batches(pairs))):
            _, metrics = objective(tts, loaded, mode, SEED + i, False)
            for k, value in metrics.items():
                totals[k] = totals.get(k, 0.) + float(value) * len(loaded)
    if not torch.isfinite(torch.tensor(list(totals.values()))).all():
        raise FloatingPointError('Nonfinite validation loss')
    return {k: v / len(pairs) for k, v in totals.items()}


def save_checkpoint(out, core, optimizer, scheduler, recipe, step, metrics):
    path = out / f'step-{step:04d}'
    with tempfile.TemporaryDirectory(dir=out, prefix='.saving-') as tmp:
        folder = Path(tmp) / 'checkpoint'
        folder.mkdir()
        full = recipe['mode'] == 'full'
        if full:
            core.save_pretrained(folder / 'fireredtts3_base', max_shard_size='100GB')
        else:
            core.save_pretrained(folder)
            (folder / 'firered_lora_metadata.json').write_text(json.dumps(dict(
                base_fingerprint=recipe['base_sha256'], step=step,
                recipe={**recipe, 'options': {'profile': 'vox_attention'}})))
        torch.save({'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                    'cpu_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all()},
                   folder / 'training_state.pt')
        (folder / 'checkpoint.json').write_text(json.dumps(dict(recipe=recipe, step=step, validation=metrics),
                                                          indent=2, allow_nan=False))
        folder.rename(path)
    return path


def transfer_recipe(mode, dataset_sha256):
    return dict(version=2, mode=mode, upstream=UPSTREAM, dataset_sha256=dataset_sha256,
                speaker_cap_seconds=CAP_SECONDS, base_sha256=BASE_SHA,
                seed=SEED, updates=TOTAL, batch_size=MICRO, accumulation=ACCUM, lr=LR[mode],
                warmup=100, weight_decay=.01, max_grad_norm=1., flow_weight=1., stop_weight=1.)


def train(dataset, output, mode, until, resume, *, fork_from=None, shared_features=None, recovery=False):
    from transformers import get_cosine_schedule_with_warmup
    if not 1 <= until <= TOTAL:
        raise ValueError('--until must be between 1 and 1000')
    if recovery and mode != 'lora':
        raise ValueError('Acoustic recovery requires LoRA')
    if recovery is True:
        recovery = vox_recovery.CONFIG
    require_training_hardware()
    print(f'[transfer] Auditing prepared audio and transcripts in {dataset}', flush=True)
    splits, rows, selection = paired_data(dataset)
    print(f'[transfer] Verified {len(rows)} unique audio files; checking the original FireRed weights', flush=True)
    fingerprint = model_identity(MODEL_DIR)
    if fingerprint != BASE_SHA:
        raise ValueError('This comparison requires the original unmerged FireRedTTS3 Base')
    parent_recipe = transfer_recipe(mode, selection['selection_sha256'])
    recipe = dict(parent_recipe, acoustic_recovery=recovery) if recovery else parent_recipe
    out = output / mode
    out.mkdir(parents=True, exist_ok=True)
    checkpoints = sorted(p for p in out.glob('step-*') if (p / 'checkpoint.json').is_file())
    if checkpoints and fork_from is not None:
        raise ValueError('Fork into a separate output without existing checkpoints')
    prior = checkpoints[-1] if checkpoints else fork_from
    saved = json.loads((prior / 'checkpoint.json').read_text()) if prior else None
    if recovery and (prior is None or 'recovery_start_step' in saved['recipe']):
        recipe['recovery_start_step'] = 0
    start = 0
    if prior:
        expected = parent_recipe if fork_from is not None else recipe
        if not resume or saved['recipe'] != expected:
            raise ValueError(f'Existing checkpoint {prior}: use --resume with the same recipe and data')
        start = saved['step']
    elif resume:
        raise ValueError('No completed checkpoint to resume')
    write_json(output / 'speaker_selection.json', selection)
    if start == until:
        print(f'[transfer] {mode} already completed {until} updates: {prior}', flush=True)
        return
    if start > until:
        raise ValueError(f'Already completed {start} updates')
    torch.manual_seed(SEED)
    manager = ModelManager()
    loader = None
    try:
        tts = manager.load_base(attention_backend='sdpa', torch_compile=False)
        tts.redae.eval().requires_grad_(False)
        tts.spk_extractor.eval().requires_grad_(False)
        cache_dir = output / 'features'
        cache = FeatureCache(tts, shared_features or cache_dir, fingerprint, True, 'native')
        if any(not cache.path(row).is_file() for row in rows):
            # A fork can read the old cache. Any new features belong to the fork.
            precompute_features(tts, rows, cache_dir, fingerprint, precision_policy='native')
            cache = FeatureCache(tts, cache_dir, fingerprint, True, 'native')
        if recovery:
            tts.redae.decoder.to('cpu')  # Keep only the frozen waveform encoder on the training device.
        else:
            tts.redae.to('cpu')
        tts.spk_extractor.to('cpu')
        torch.cuda.empty_cache()
        loader = BatchFeatures(cache)
        core = tts.tts_core = configure(tts.tts_core, mode, prior)
        core.backbone_llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        params = [p for p in core.parameters() if p.requires_grad]
        if torch.device(tts.device).type == "cuda":
            torch.cuda.reset_peak_memory_stats(tts.device)
        optimizer = torch.optim.AdamW(params, lr=LR[mode], weight_decay=.01)
        scheduler = get_cosine_schedule_with_warmup(optimizer, 100, TOTAL)
        if prior:
            state = torch.load(prior / 'training_state.pt', map_location='cpu', weights_only=True)
            optimizer.load_state_dict(state['optimizer'])
            scheduler.load_state_dict(state['scheduler'])
            torch.set_rng_state(state['cpu_rng'])
            torch.cuda.set_rng_state_all(state['cuda_rng'])
            del state
        print(f'[transfer] {mode}, trainable={sum(p.numel() for p in params):,}, updates {start+1}..{until}',
              flush=True)
        stream = loader.iterate(batches(splits['train'][start*MICRO*ACCUM:until*MICRO*ACCUM]))
        if not prior:
            metrics = validate(tts, loader, splits['validation'], mode)
            print(f'[validation] base step=0 {metrics}', flush=True)
        set_training(core)
        for step in range(start+1, until+1):
            optimizer.zero_grad(set_to_none=True)
            loss_sum = 0.
            recovery_sum, windows = 0., 0
            patch_counts = [0] * recovery['prediction_patches'] if recovery and 'prediction_patches' in recovery else None
            for micro in range(ACCUM):
                _, loaded = next(stream)
                seed = SEED + (step-1)*ACCUM + micro
                loss, _ = objective(tts, loaded, mode, seed, True)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'Nonfinite {mode} loss at update {step}, microbatch {micro+1}')
                (loss / ACCUM).backward()
                loss_sum += float(loss.detach()) / ACCUM
                if recovery:
                    # Backward separately so clean and recovery activations do not coexist.
                    examples = to_examples(tts, loaded, 'native')
                    with isolated_rng(seed + 100_000):
                        extra, count = vox_recovery.recovery_loss(tts, examples, [r[3] for r in loaded],
                                                                  seed + 100_000, config=recovery, patch_counts=patch_counts)
                        if not torch.isfinite(extra):
                            raise FloatingPointError(f'Nonfinite recovery loss at update {step}')
                        if count:
                            (recovery['weight'] * extra / ACCUM).backward()
                    recovery_sum += float(extra.detach()) / ACCUM
                    windows += count
            if recovery:
                counts = f' targets_by_hop={patch_counts}' if patch_counts is not None else ''
                print(f'[recovery] step={step} windows={windows} clean_loss={loss_sum:.6f} '
                      f'recovery_flow={recovery_sum:.6f}{counts}', flush=True)
                loss_sum += recovery['weight'] * recovery_sum
            norm = torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            if step % 10 == 0 or step == until:
                print(f'[transfer] {mode} {step}/{TOTAL} loss={loss_sum:.6f} grad_norm={norm:.4f}', flush=True)
            if step % 500 == 0 or step == until:
                metrics = validate(tts, loader, splits['validation'], mode)
                if torch.device(tts.device).type == 'cuda':
                    print(f'[transfer] Peak allocated GPU memory: {torch.cuda.max_memory_allocated(tts.device)/2**30:.2f} GiB',
                          flush=True)
                print(f'[transfer] Saving {mode} at step {step}', flush=True)
                path = save_checkpoint(out, core, optimizer, scheduler, recipe, step, metrics)
                print(f'[validation] {metrics}; checkpoint={path}', flush=True)
                set_training(core)
    finally:
        if loader:
            loader.close()
        manager.unload()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('selection', help='Audit the data and report the longest-first training selection on CPU')
    p = sub.add_parser('train')
    p.add_argument('mode', choices=LR)
    p.add_argument('--until', type=int, default=TOTAL, help='Stop after this update without changing the LR schedule')
    p.add_argument('--resume', action='store_true', help='Continue from the last complete checkpoint for this mode')
    p = sub.add_parser('compare')
    p.add_argument('--step', type=int, default=TOTAL)
    p.add_argument('--lora-only', action='store_true', help='Generate with the original base and the LoRA checkpoint')
    bench = p.add_mutually_exclusive_group()
    bench.add_argument('--benchmark', type=Path, help='Optional JSONL with reference audio, text and seeds')
    bench.add_argument('--against', type=Path, help='Replay a report from an earlier step of the same recipe or from the original data at the same step')
    p.add_argument('--asr', default='large-v3')
    p.add_argument('--speaker-model', default='microsoft/wavlm-base-plus-sv')
    for p in sub.choices.values():
        p.add_argument('--dataset', type=Path, default=Path(voxcpm_data.FAIR_OUTPUT))
        p.add_argument('--output', type=Path, default=Path(OUTPUT))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with FileLock(str(args.output / '.run.lock'), timeout=0):
        if args.command == 'selection':
            _, _, selection = paired_data(args.dataset)
            path = args.output / 'speaker_selection.json'
            write_json(path, selection)
            print(f"{selection['balance']['unique_targets']} training clips, {selection['train_hours']:.2f} h, "
                  f"{selection['validation_clips']} validation clips. Report: {path}")
        elif args.command == 'train':
            train(args.dataset, args.output, args.mode, args.until, args.resume)
        else:
            from .vox_recipe_compare import compare
            print(compare(args.dataset, args.output, args.step, args.benchmark, args.asr, args.speaker_model,
                          args.against, lora_only=args.lora_only))


if __name__ == '__main__':
    main()

"""Neutral short Dialogs: reproducible thirds, session holdout and audio audit."""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path, PurePosixPath
import random
import re

from .training_data import (SpeakerPairs, assert_disjoint, file_digest, load_manifest,
                            plain_text, write_manifest)
from .yo_dataset import has_sentence_ending

REPO = 'langswap/dialogs-ru-emotional-conversations'
REVISION = 'a74ff44d8d27ca3d21098ea3ed4d31a32923fbdd'
METADATA = ('metadata.csv', 'train.csv', 'val.csv', 'test.csv', 'LICENSE.md')
SPEAKERS = {'D': 'Дима', 'M': 'Маша', 'S': 'Света'}


def audio_name(value):
    path = PurePosixPath(value)
    if ('\\' in value or path.is_absolute() or '..' in path.parts or len(path.parts) != 2
            or path.parts[0] != 'wavs' or not re.fullmatch(r'[a-z0-9_]+_\d+\.wav', path.name)):
        raise ValueError(f'Invalid Dialogs audio path: {value!r}')
    return path.as_posix()


def text_key(text):
    # Keep punctuation contrasts ("Готово." / "Готово?") within training.
    return plain_text(text).casefold().replace('ё', 'е')


def lexical_key(text):
    # Across train/validation, also exclude the same words with different marks.
    return ' '.join(re.findall(r'\w+', text_key(text)))


def category(text):
    if '?' in text:
        return 'question'
    ending = re.sub(r'[\s»”’"\)\]\}]+$', '', text)
    if '!' not in text and ending.endswith('.'):
        return 'statement'
    return 'other'


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream, delimiter='|'))
    seen = set()
    for row in rows:
        name = audio_name(row['audio_path'])
        if name in seen:
            raise ValueError(f'Duplicate audio path in {path}: {name}')
        seen.add(name)
    return rows


def eligible(rows, max_seconds=10., available_files=None):
    result, rejected = [], Counter()
    for row in rows:
        try:
            if row['emotion'].strip() != 'neutral':
                raise ValueError('not_neutral')
            if row['speaker_id'] not in SPEAKERS:
                raise ValueError('unknown_speaker')
            duration = float(row['duration'])
            if not math.isfinite(duration) or not 0 < duration <= max_seconds:
                raise ValueError('duration')
            text = plain_text(row['text'])
            if not lexical_key(text) or not has_sentence_ending(text):
                raise ValueError('incomplete_sentence_ending')
            name = audio_name(row['audio_path'])
            if available_files is not None and name not in available_files:
                raise ValueError('missing_audio')
            result.append(dict(audio_path=name, speaker_id=row['speaker_id'], text=text,
                               duration=duration, emotion='neutral', category=category(text),
                               session=re.sub(r'_\d+\.wav$', '', PurePosixPath(name).name)))
        except (ValueError, KeyError) as exc:
            rejected[str(exc)] += 1
    return result, dict(rejected)


def summarize(rows):
    return {'clips': len(rows), 'minutes': sum(r['duration'] for r in rows)/60,
            'categories': dict(Counter(r['category'] for r in rows)),
            'slots': dict(Counter(r['selection_slot'] for r in rows if 'selection_slot' in r)),
            'sessions': len({r['session'] for r in rows}),
            'length_bins': dict(Counter(str(min(4, int(r['duration']//2))) for r in rows))}


def diverse_order(rows, seed):
    rng = random.Random(seed)
    shuffled = sorted(rows, key=lambda r: r['audio_path'])
    rng.shuffle(shuffled)
    groups = defaultdict(list)
    seen = set()
    for row in shuffled:
        key = text_key(row['text'])
        if key in seen:
            continue
        seen.add(key)
        groups[(row['session'], min(4, int(row['duration']//2)))].append(row)
    keys = sorted(groups)
    rng.shuffle(keys)
    ordered = []
    while keys:
        for key in keys:
            ordered.append(groups[key].pop())
        keys = [key for key in keys if groups[key]]
    return ordered


def choose_thirds(rows, seconds, seed):
    """Exact count thirds; random slot draws from ALL still-unused candidates.

    Duration is a cap on whole clips, never a crop. Fill in triples so the
    ratio remains exact even when the time budget cannot fit another triple.
    """
    order = diverse_order(rows, seed)
    selected, used = [], set()
    remaining = seconds
    while True:
        rest = [r for r in order if r['audio_path'] not in used]
        questions = [r for r in rest if r['category'] == 'question']
        statements = [r for r in rest if r['category'] == 'statement']
        if len(rest) < 3 or not questions or not statements:
            break
        min_random = min(r['duration'] for r in rest)
        min_statement = min(r['duration'] for r in statements)
        triple = None
        for question in questions:
            if question['duration'] + min_statement + min_random > remaining + 1e-8:
                continue
            for statement in statements:
                room = remaining - question['duration'] - statement['duration']
                if room + 1e-8 < min_random:
                    continue
                random_row = next((r for r in rest if r['audio_path'] not in
                                   {question['audio_path'], statement['audio_path']}
                                   and r['duration'] <= room + 1e-8), None)
                if random_row:
                    triple = (question, statement, random_row)
                    break
            if triple:
                break
        if triple is None:
            break
        number = len(selected)//3
        for slot, row in zip(('question', 'statement', 'random'), triple):
            selected.append({**row, 'selection_slot': slot, 'selection_triple': number})
            used.add(row['audio_path'])
            remaining -= row['duration']
    return selected


def select_with_holdout(rows, train_seconds, validation_seconds, seed, max_seconds):
    """Reserve complete original sessions, then sample extra validation minutes.

    All actors occur in both splits: this is a recording/text holdout. Unseen
    voice validation stays in YO; actor-disjoint Dialogs would lose an actor.
    """
    sources = sorted({r['session'] for r in rows})
    for attempt in range(64):
        order = sources.copy()
        random.Random(f'dialogs-holdout-v1:{seed}:{attempt}').shuffle(order)
        heldout, validation = set(), {}
        for session in order:
            deficient = {s for s in SPEAKERS if sum(r['duration'] for r in validation.get(s, []))
                         < validation_seconds-max_seconds}
            if not deficient:
                break
            if not any(r['session'] == session and r['speaker_id'] in deficient for r in rows):
                continue
            heldout.add(session)
            validation = {s: choose_thirds([r for r in rows if r['session'] in heldout and r['speaker_id'] == s],
                          validation_seconds, f'{seed}:validation:{s}') for s in SPEAKERS}
        if any(sum(r['duration'] for r in validation.get(s, [])) < validation_seconds-max_seconds for s in SPEAKERS):
            continue
        texts = {lexical_key(r['text']) for selected in validation.values() for r in selected}
        train = {s: choose_thirds([r for r in rows if r['session'] not in heldout and r['speaker_id'] == s
                                 and lexical_key(r['text']) not in texts],
                                 train_seconds, f'{seed}:train:{s}') for s in SPEAKERS}
        if all(sum(r['duration'] for r in train[s]) >= train_seconds-max_seconds for s in SPEAKERS):
            return train, validation, sorted(heldout)
    raise ValueError('Cannot fill both per-speaker budgets with disjoint sessions and balanced thirds. '
                     'Reduce the budgets or inspect the metadata; no speaker/quota was silently dropped.')


def fetch_files(source, names, revision):
    from concurrent.futures import ThreadPoolExecutor
    from huggingface_hub import hf_hub_download
    def fetch(name):
        return hf_hub_download(REPO, name, repo_type='dataset', revision=revision, local_dir=source)
    with ThreadPoolExecutor(max_workers=8) as workers:
        list(workers.map(fetch, names))


def file_inventory(source, revision, refresh=False):
    """Cache the full paginated file tree, without downloading audio payloads.

    CSV rows are not evidence that a WAV was uploaded. Keep the inventory tied
    to the same immutable revision and fail on API errors, not an empty list.
    """
    if not re.fullmatch(r'[0-9a-f]{40}', revision):
        raise ValueError('Pin Dialogs to a full commit SHA')
    path = Path(source)/f'file-inventory-{revision}.json'
    if refresh:
        from huggingface_hub import HfApi
        # list_repo_files consumes every page; do not use a preview/file sample.
        files = HfApi().list_repo_files(REPO, repo_type='dataset', revision=revision)
        data = dict(schema=1, repo=REPO, revision=revision, files=sorted(files))
    else:
        if not path.is_file():
            raise ValueError('Missing pinned file inventory; run plan-dialogs with --download first')
        data = json.loads(path.read_text(encoding='utf-8'))
    files = data.get('files')
    if (data.get('schema') != 1 or data.get('repo') != REPO or data.get('revision') != revision
            or not isinstance(files, list) or not all(isinstance(name, str) for name in files)
            or len(set(files)) != len(files) or not set(METADATA).issubset(files)):
        raise ValueError('Invalid pinned file inventory; regenerate with plan-dialogs --download')
    if refresh:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
        temporary.replace(path)
    return set(files), file_digest(path)


def plan_dialogs(source, output, revision=REVISION, max_seconds=10., train_minutes=30.,
                 validation_minutes=10/3, seed=1234, download=False):
    source, output = Path(source).resolve(), Path(output).resolve()
    if not re.fullmatch(r'[0-9a-f]{40}', revision):
        raise ValueError('Pin Dialogs to a full commit SHA')
    if not all(math.isfinite(v) and v > 0 for v in (max_seconds, train_minutes, validation_minutes)):
        raise ValueError('Durations and budgets must be positive finite numbers')
    if min(train_minutes, validation_minutes)*60 <= max_seconds:
        raise ValueError('Each budget must exceed one maximum-length utterance')
    if output.exists() and any(output.iterdir()):
        raise ValueError('Plan output must be new/empty')
    if download:
        fetch_files(source, METADATA, revision)
    files = {name: read_csv(source/name) for name in METADATA if name.endswith('.csv')}
    canonical = {r['audio_path']: r for r in files['metadata.csv']}
    seen = set()
    for name in ('train.csv', 'val.csv', 'test.csv'):
        for row in files[name]:
            if row['audio_path'] in seen or canonical.get(row['audio_path']) != row:
                raise ValueError('Official split overlap or mismatch with metadata.csv')
            seen.add(row['audio_path'])
    if seen != canonical.keys():
        raise ValueError('Official splits do not cover metadata.csv exactly')
    available, inventory_hash = file_inventory(source, revision, refresh=download)
    all_rows, all_rejected = eligible(files['metadata.csv'], max_seconds, available)
    pool, rejected = eligible(files['train.csv'], max_seconds, available)
    train, validation, heldout = select_with_holdout(pool, train_minutes*60, validation_minutes*60, seed, max_seconds)
    report = {'repo': REPO, 'revision': revision, 'max_seconds': max_seconds, 'emotion': 'neutral',
              'seed': seed, 'train_minutes_per_speaker': train_minutes,
              'validation_minutes_per_speaker': validation_minutes, 'heldout_sessions': heldout,
              'all_metadata_rejections': all_rejected, 'official_train_rejections': rejected,
              'file_inventory': {'repo_files': len(available),
                                 'metadata_missing_audio': len(canonical.keys()-available),
                                 'sha256': inventory_hash},
              'random_slot': 'All remaining eligible recordings; no repeated recording or text within a speaker/split.',
              'speakers': {s: {'name': SPEAKERS[s],
                             'all_metadata_eligible': summarize([r for r in all_rows if r['speaker_id'] == s]),
                             'official_train_eligible': summarize([r for r in pool if r['speaker_id'] == s]),
                             'train': summarize(train[s]), 'validation': summarize(validation[s])} for s in SPEAKERS},
              'note': 'Metadata estimates after checking the pinned file tree, before waveform audit. Dialogs validation has known actors but '
                      'disjoint sessions and word sequences; keep separate unseen-speaker YO validation.'}
    plan = {'schema': 1, 'source_dir': str(source), 'report': report,
            'source_hashes': {name: file_digest(source/name) for name in METADATA},
            'file_inventory_sha256': inventory_hash,
            'train': [r for s in SPEAKERS for r in train[s]],
            'validation': [r for s in SPEAKERS for r in validation[s]]}
    output.mkdir(parents=True, exist_ok=True)
    (output/'plan.json').write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding='utf-8')
    (output/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    return report


def short_support_rows(path, max_seconds):
    """Keep complete <=10s YO clips and whole groups with valid references."""
    source = load_manifest(path)
    rows = [r for r in source if r['duration'] <= max_seconds and has_sentence_ending(r['text'])]
    groups = defaultdict(list)
    for row in rows:
        groups[(row['speaker_id'], row['language'])].append(row)
    kept = [row for clips in groups.values() for row in clips
            if any(SpeakerPairs.distinct(row, ref) for ref in clips)]
    SpeakerPairs(kept, max_seconds, max_seconds)
    return kept, {'source_clips': len(source), 'selected_clips': len(kept),
                  'speakers': len({r['speaker_id'] for r in kept}),
                  'minutes': sum(r['duration'] for r in kept)/60,
                  'source_manifest_sha256': file_digest(path)}


def validate_plan(plan):
    maximum = plan['report']['max_seconds']
    for split, key in [('train', 'train_minutes_per_speaker'), ('validation', 'validation_minutes_per_speaker')]:
        groups = defaultdict(list)
        for row in plan[split]:
            if row['selection_slot'] not in {'question', 'statement', 'random'}:
                raise ValueError('Invalid selection slot')
            if row['selection_slot'] != 'random' and row['selection_slot'] != row['category']:
                raise ValueError('Selection slot does not match punctuation')
            groups[row['speaker_id']].append(row)
        if set(groups) != set(SPEAKERS):
            raise ValueError('Plan must contain all three actors in each split')
        for rows in groups.values():
            if len({text_key(r['text']) for r in rows}) != len(rows):
                raise ValueError('Repeated text within actor/split')
            if sum(r['duration'] for r in rows) > plan['report'][key]*60 + 1e-6:
                raise ValueError('Plan exceeds its per-speaker duration budget')
            triples = defaultdict(list)
            for row in rows:
                triples[row['selection_triple']].append(row)
                if not 0 < row['duration'] <= maximum:
                    raise ValueError('Plan contains an overlong recording')
            for triple in triples.values():
                if len(triple) != 3 or {r['selection_slot'] for r in triple} != {'question','statement','random'}:
                    raise ValueError('Plan does not contain exact thirds')
    if {r['session'] for r in plan['train']} & {r['session'] for r in plan['validation']}:
        raise ValueError('Plan train/validation session leakage')
    if {lexical_key(r['text']) for r in plan['train']} & {lexical_key(r['text']) for r in plan['validation']}:
        raise ValueError('Plan train/validation text leakage')


def prepare_dialogs(plan_path, output, download=False, log=print, replay_manifest=None, validation_manifest=None):
    import numpy as np
    import soundfile as sf
    import torch
    import torchaudio
    plan = json.loads(Path(plan_path).read_text(encoding='utf-8'))
    if plan.get('schema') != 1 or plan['report']['repo'] != REPO:
        raise ValueError('Unsupported Dialogs plan')
    root, source = Path(output).resolve(), Path(plan['source_dir']).resolve()
    if root.exists() and any(root.iterdir()):
        raise ValueError('Import output must be new/empty')
    if bool(replay_manifest) != bool(validation_manifest):
        raise ValueError('Supply both YO replay and unseen validation manifests')
    for name in METADATA:
        if file_digest(source/name) != plan['source_hashes'].get(name):
            raise ValueError(f'Source metadata changed: {name}; regenerate the plan')
    selected = plan['train'] + plan['validation']
    if len({r['audio_path'] for r in selected}) != len(selected):
        raise ValueError('Repeated audio in Dialogs plan')
    validate_plan(plan)
    # Bind plan entries to the source metadata; do not trust edited transcripts,
    # speaker IDs, durations or memberships in a manually modified plan.
    official, _ = eligible(read_csv(source/'train.csv'), plan['report']['max_seconds'])
    by_name = {r['audio_path']: r for r in official}
    for row in selected:
        if any(row.get(k) != v for k, v in by_name.get(row['audio_path'], {}).items()) or row['audio_path'] not in by_name:
            raise ValueError('Plan row differs from official eligible metadata')
        audio_name(row['audio_path'])
    # Reject stale CSV-only plans before loading YO or downloading any WAVs.
    # Legacy plans remain usable if every selected file actually exists.
    if download or plan.get('file_inventory_sha256'):
        available, inventory_hash = file_inventory(source, plan['report']['revision'], refresh=download)
        if plan.get('file_inventory_sha256') and inventory_hash != plan['file_inventory_sha256']:
            raise ValueError('File inventory changed; regenerate the plan')
        missing = sorted(r['audio_path'] for r in selected if r['audio_path'] not in available)
        if missing:
            raise ValueError(f'Plan selects {len(missing)} WAVs absent from the pinned repository; '
                             f'first: {missing[:3]}. Rerun plan-dialogs --download with a new output '
                             'directory, then import the new plan. Keep existing downloaded WAVs.')
    support, support_reports = {}, {}
    if replay_manifest:
        for name, path in [('replay', replay_manifest), ('unseen-validation', validation_manifest)]:
            support[name], support_reports[name] = short_support_rows(path, plan['report']['max_seconds'])
        assert_disjoint(support['replay'], support['unseen-validation'], 'speaker')
    if download:
        fetch_files(source, [r['audio_path'] for r in selected], plan['report']['revision'])
    (root/'audio').mkdir(parents=True, exist_ok=True)
    accepted, rejected, hashes = {}, {}, set()
    for index, row in enumerate(selected, 1):
        path = (source/row['audio_path']).resolve()
        if not path.is_relative_to(source):
            raise ValueError('Audio symlink escapes source directory')
        if not path.is_file():
            raise FileNotFoundError(f'Missing {path}; run prepare-dialogs with --download')
        try:
            audio, rate = sf.read(path, dtype='float32', always_2d=True)
            duration = len(audio)/rate
            if audio.shape[1] != 1 or not len(audio) or not np.isfinite(audio).all():
                raise ValueError('invalid_waveform')
            if abs(duration-row['duration']) > .15:
                raise ValueError('duration_mismatch')
            if duration > plan['report']['max_seconds']:
                raise ValueError('actual_duration_over_limit')
            if (np.abs(audio) >= .999).mean() > .001:
                raise ValueError('clipping')
            waveform = torch.from_numpy(audio[:, 0].copy())
            if rate != 24000:
                waveform = torchaudio.functional.resample(waveform, rate, 24000)
            if len(waveform)/24000 > plan['report']['max_seconds']:
                raise ValueError('resampled_duration_over_limit')
            destination = root/'audio'/Path(row['audio_path']).name
            sf.write(destination, waveform.numpy(), 24000, subtype='PCM_16')
            digest = file_digest(destination)
            if digest in hashes:
                destination.unlink()
                raise ValueError('duplicate_audio')
            hashes.add(digest)
            accepted[row['audio_path']] = {**row, 'audio': str(destination), 'audio_sha256': digest,
                'utt_id': 'dialogs:'+row['audio_path'], 'speaker_id': 'dialogs:'+row['speaker_id'],
                'global_spk_id': 'dialogs:'+row['speaker_id'], 'language': 'Russian',
                'source_id': 'dialogs:'+row['session'], 'segment_source_id': 'dialogs:'+row['audio_path'],
                'start': 0., 'end': len(waveform)/24000, 'duration': len(waveform)/24000,
                'source_revision': plan['report']['revision']}
        except (ValueError, RuntimeError) as exc:
            rejected[row['audio_path']] = str(exc)
        if index % 100 == 0 or index == len(selected):
            log(f'[dialogs] Audio audit {index}/{len(selected)}; rejected={len(rejected)}')
    splits, discarded = {}, []
    for split in ('train', 'validation'):
        triples = defaultdict(list)
        for row in plan[split]:
            triples[(row['speaker_id'], row['selection_triple'])].append(row)
        splits[split] = []
        for rows in triples.values():
            if len(rows) != 3 or {r['selection_slot'] for r in rows} != {'question', 'statement', 'random'}:
                raise ValueError('Plan does not contain exact thirds')
            if all(r['audio_path'] in accepted for r in rows):
                splits[split].extend(accepted[r['audio_path']] for r in rows)
            else:
                discarded.extend(r['audio_path'] for r in rows)
    for name in discarded:
        if name in accepted:
            Path(accepted[name]['audio']).unlink()
    report = {'plan_sha256': file_digest(plan_path), 'source': plan['report'], 'audio_rejections': rejected,
              'support': support_reports,
              'discarded_triple_members': discarded, 'splits': {split: {
                  s: summarize([r for r in rows if r['speaker_id'] == 'dialogs:'+s]) for s in SPEAKERS}
                  for split, rows in splits.items()},
              'note': 'Whole triples removed after an audio rejection to preserve count thirds. '
                      'No MOS or word alignment is supplied by Dialogs; transcript/listening audit is still needed.'}
    (root/'import_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    for split, rows in splits.items():
        if {r['speaker_id'] for r in rows} != {'dialogs:'+s for s in SPEAKERS}:
            raise ValueError('An actor is missing after audio audit; see import_report.json')
        SpeakerPairs(rows, plan['report']['max_seconds'], plan['report']['max_seconds'])
    assert_disjoint(splits['train'], splits['validation'], 'recording')
    if {lexical_key(r['text']) for r in splits['train']} & {lexical_key(r['text']) for r in splits['validation']}:
        raise ValueError('Train/validation text leakage')
    if support:
        assert_disjoint(splits['train'] + splits['validation'] + support['replay'], support['unseen-validation'], 'speaker')
        assert_disjoint(splits['validation'], support['replay'], 'recording')
    for split, rows in splits.items():
        write_manifest(root/f'{split}.jsonl', rows)
    for name, rows in support.items():
        write_manifest(root/f'{name}.jsonl', rows)
    return report

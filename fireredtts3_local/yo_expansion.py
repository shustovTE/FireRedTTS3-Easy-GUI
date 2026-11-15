"""Authenticated, resumable expansion of YO from a fixed selection of Parquet shards."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import tempfile

from .hub_download import download_files, local_target
from .training_data import (SpeakerPairs, assert_disjoint, dataset_fingerprint, file_digest,
                            load_manifest, split_with_anchors, write_manifest)
from .yo_dataset import import_yo

REPO = 'NCSpeech/YO-CPT-ru'
REVISION = 'f2f92c5ecf6b0323cf7e7f9599f0e4852f6c536b'
TRAIN = ['training/datasets/yo-split/train.jsonl', 'training/datasets/yo-complete-split/train.jsonl',
         'training/datasets/dialogs-neutral/train.jsonl']
VALIDATION = ['training/datasets/yo-split/validation.jsonl', 'training/datasets/yo-complete-split/validation.jsonl']
PROTECTED = ['training/datasets/dialogs-neutral/validation.jsonl']


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def checked_rows(path):
    """Check existing WAV content against the recorded hash, not only existence."""
    path = Path(path).resolve()
    original = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    if not original:
        return []
    rows = load_manifest(path)
    for old, actual in zip(original, rows):
        if old.get('audio_sha256') != actual['audio_sha256']:
            raise ValueError(f'Missing/mismatched recorded WAV checksum in {path}: {actual["utt_id"]}')
    return rows


def check_anchors(plan):
    groups = {}
    for role, files in plan['anchors'].items():
        rows = []
        for entry in files:
            if file_digest(entry['path']) != entry['sha256']:
                raise ValueError('Historical manifest changed; make a new expansion plan')
            loaded = checked_rows(entry['path'])
            if dataset_fingerprint(loaded) != entry['fingerprint']:
                raise ValueError('Historical WAVs changed; do not remove original Parquet')
            rows.extend(loaded)
        groups[role] = rows
    assert_disjoint(groups['train'], groups['validation'], 'speaker')
    return groups


def validate_shard(entry):
    if (not re.fullmatch(r'data/ru\d+-\d+\.parquet', entry['path'])
            or type(entry['size']) is not int or entry['size'] <= 0
            or not re.fullmatch(r'[0-9a-f]{64}', entry['sha256'])):
        raise ValueError('Invalid pinned Parquet entry')


def check_shard(source, entry):
    validate_shard(entry)
    path = local_target(source, entry['path'])
    if (path.is_symlink() or not path.is_file() or path.stat().st_size != entry['size']
            or file_digest(path) != entry['sha256']):
        raise ValueError(f'Parquet size/checksum mismatch: {entry["path"]}')
    return path


def used_shards(source, revision):
    path = Path(source)/'used-parquet.json'
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    if data.get('repo') != REPO or data.get('revision') != revision:
        raise ValueError('Used-shard ledger belongs to a different dataset revision')
    return {entry['path']: entry for entry in data['files']}


def register_used(source, revision, entries):
    previous = used_shards(source, revision)
    for entry in entries:
        validate_shard(entry)
        if entry['path'] in previous and previous[entry['path']] != entry:
            raise ValueError('Used-shard checksum changed')
        previous[entry['path']] = entry
    write_json(Path(source)/'used-parquet.json', dict(repo=REPO, revision=revision,
               files=[previous[name] for name in sorted(previous)]))


def choose_shards(catalog, excluded, fraction=.5, seed=1234):
    """Additional half by shard count, proportional across original ru* parts."""
    if not math.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError('fraction must be in (0, 1]')
    count = math.ceil(len(catalog)*fraction)
    groups = defaultdict(list)
    for entry in sorted(catalog, key=lambda r:r['path']):
        if entry['path'] not in excluded:
            groups[Path(entry['path']).stem.split('-')[0]].append(entry)
    available = sum(map(len, groups.values()))
    if count > available:
        raise ValueError(f'Need {count} new shards, but only {available} remain; reduce fraction')
    exact = {part:count*len(rows)/available for part,rows in groups.items()}
    quotas = {part:math.floor(value) for part,value in exact.items()}
    for part in sorted(groups, key=lambda p:(-(exact[p]-quotas[p]),p))[:count-sum(quotas.values())]:
        quotas[part] += 1
    selected = []
    rng = random.Random(seed)
    for part in sorted(groups):
        rng.shuffle(groups[part])
        selected.extend(groups[part][:quotas[part]])
    rng.shuffle(selected)
    return selected


def plan_expansion(source, output, fraction=.5, seed=1234, revision=REVISION,
                   train_anchors=TRAIN, validation_anchors=VALIDATION, protected_manifests=PROTECTED):
    from huggingface_hub import HfApi, get_token
    if not get_token():
        raise ValueError('Run .venv/bin/hf auth login on this machine before planning')
    if not re.fullmatch(r'[0-9a-f]{40}', revision):
        raise ValueError('Pin YO to a full commit SHA')
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError('Plan output must be new/empty')
    info = HfApi(token=True).dataset_info(REPO, revision=revision, files_metadata=True)
    if info.sha != revision:
        raise ValueError('Hub returned a different revision')
    catalog = []
    for item in info.siblings:
        if item.rfilename.endswith('.parquet'):
            entry = dict(path=item.rfilename, size=item.size, sha256=item.lfs.sha256)
            validate_shard(entry)
            catalog.append(entry)
    if not catalog or len({r['path'] for r in catalog}) != len(catalog):
        raise ValueError('Invalid Hub shard inventory')
    by_path = {r['path']:r for r in catalog}
    previous = used_shards(source, revision)
    if any(by_path.get(name) != entry for name,entry in previous.items()):
        raise ValueError('Used-shard ledger does not match pinned Hub inventory')
    old = []
    for path in sorted((source/'data').glob('*.parquet')):
        name = path.relative_to(source).as_posix()
        if name not in by_path:
            raise ValueError(f'Unrecognized local Parquet: {name}')
        check_shard(source, by_path[name])
        old.append(by_path[name])
    selected = choose_shards(catalog, set(previous) | {r['path'] for r in old}, fraction, seed)
    anchors = {}
    for role, paths in [('train',train_anchors),('validation',validation_anchors),('protected',protected_manifests)]:
        anchors[role] = []
        for path in paths:
            path = Path(path).resolve()
            rows = checked_rows(path)
            if not rows:
                raise ValueError(f'Empty historical manifest: {path}')
            anchors[role].append(dict(path=str(path),sha256=file_digest(path),fingerprint=dataset_fingerprint(rows)))
    if not anchors['train'] or not anchors['validation']:
        raise ValueError('Both historical train and validation anchors are required')
    # Small metadata requests also use the stored token; never save it in a plan.
    download_files(source,['README.md','LICENSE.md','shard_map.tsv'],REPO,revision,authenticated=True)
    with (source/'shard_map.tsv').open() as stream:
        hours = {r['shard']:float(r['hours']) for r in csv.DictReader(stream,delimiter='\t')}
    original_hours = sum(hours[Path(r['path']).stem] for r in selected)
    report = dict(total_shards=len(catalog),total_bytes=sum(r['size'] for r in catalog),
                  selected_new_shards=len(selected),selected_bytes=sum(r['size'] for r in selected),
                  source_hours_before_filtering=original_hours,
                  old_parquet_files=len(old),reclaimable_bytes=sum(r['size'] for r in old),
                  parts=dict(Counter(Path(r['path']).stem.split('-')[0] for r in selected)),
                  fraction_definition='ceil(fraction * all shard count), excluding previously downloaded/used shards',
                  pcm_upper_estimate_bytes=math.ceil(original_hours*3600*24000*2*1.1))
    report['free_bytes_now'] = shutil.disk_usage(source).free
    report['estimated_peak_extra_bytes'] = report['selected_bytes']+report['pcm_upper_estimate_bytes']
    plan = dict(schema=1,repo=REPO,revision=revision,source_dir=str(source),seed=seed,
                selected=selected,old_parquet=old,anchors=anchors,report=report,
                filters=dict(segmentation='complete-utterance',min_mos=4.,max_hours=0.,max_speaker_minutes=30.))
    check_anchors(plan)
    write_json(output/'plan.json',plan)
    write_json(output/'report.json',report)
    return report


def remove_old(plan):
    """Unlink only the explicitly inventoried raw files, after WAV/hash checks."""
    check_anchors(plan)
    source = Path(plan['source_dir'])
    selected = {r['path'] for r in plan['selected']}
    if selected & {r['path'] for r in plan['old_parquet']}:
        raise ValueError('Old/new shard overlap; refusing cleanup')
    present = [entry for entry in plan['old_parquet'] if local_target(source,entry['path']).exists()]
    # Check every candidate before deleting the first one.
    paths = [check_shard(source,entry) for entry in present]
    register_used(source, plan['revision'], plan['old_parquet'])
    for path in paths:
        path.unlink()
    return dict(deleted_files=len(paths),deleted_bytes=sum(r['size'] for r in present))


def raw_utterances(rows):
    result = set()
    for row in rows:
        name = row.get('segment_source_id','')
        if name.startswith('yo:'):
            result.add(name[3:])
        elif row['utt_id'].startswith('yo:'):
            result.add(row['utt_id'][3:].rsplit(':',1)[0])
    return result


def prune_unused(root, kept):
    """Remove only checksum-bound extraction rejects after final manifests exist."""
    record = root/'unused_audio.json'
    if not record.is_file():
        return
    retained = {Path(r['audio']).resolve() for r in kept}
    pending = []
    for entry in json.loads(record.read_text()):
        path = local_target(root,entry['path'])
        if (path.suffix != '.wav' or not path.resolve().is_relative_to(root/'parts')
                or path.resolve() in retained or path.is_symlink()):
            raise ValueError('Invalid unused-audio cleanup entry')
        if path.exists():
            if file_digest(path) != entry['sha256']:
                raise ValueError('Unused extracted WAV changed; refusing cleanup')
            pending.append(path)
    for path in pending:
        path.unlink()


def run_expansion(plan_path, output, delete_old=False, transport='xet', log=print):
    from filelock import FileLock
    plan_path = Path(plan_path).resolve()
    plan = json.loads(plan_path.read_text())
    if plan.get('schema') != 1 or plan.get('repo') != REPO:
        raise ValueError('Unsupported YO expansion plan')
    source, root = Path(plan['source_dir']).resolve(), Path(output).resolve()
    fingerprint = file_digest(plan_path)
    marker = root/'expansion_state.json'
    if root.exists() and any(root.iterdir()):
        if not marker.is_file() or json.loads(marker.read_text()).get('plan_sha256') != fingerprint:
            raise ValueError('Output belongs to a different plan or an unrelated import')
    root.mkdir(parents=True, exist_ok=True)
    with FileLock(str(source/'.yo-expansion.lock'),timeout=0):
        status = dict(plan_sha256=fingerprint,status='running',phase='preflight',complete=False,pid=os.getpid())
        write_json(marker,status)
        try:
            groups = check_anchors(plan)
            if (root/'split_report.json').is_file():
                train, validation = checked_rows(root/'train.jsonl'), checked_rows(root/'validation.jsonl')
                assert_disjoint(train+groups['train'],validation+groups['validation'],'speaker')
                report = json.loads((root/'split_report.json').read_text())
                if report['plan_sha256'] != fingerprint:
                    raise ValueError('Final report belongs to a different plan')
                prune_unused(root,train+validation)
                status.update(status='completed',phase='complete',complete=True)
                write_json(marker,status)
                return report
            for entry in plan['selected']+plan['old_parquet']:
                validate_shard(entry)
            if len({r['path'] for r in plan['selected']}) != len(plan['selected']):
                raise ValueError('Repeated selected shard')
            if {r['path'] for r in plan['selected']} & {r['path'] for r in plan['old_parquet']}:
                raise ValueError('Selected/old shard overlap')
            missing_bytes = sum(r['size'] for r in plan['selected'] if not local_target(source,r['path']).is_file())
            extracted_bytes = sum(p.stat().st_size for p in (root/'parts').rglob('*.wav'))
            pcm_remaining = max(0,plan['report'].get('pcm_upper_estimate_bytes',0)-extracted_bytes)
            headroom = max(2*1024**3,2*max(r['size'] for r in plan['selected']))
            reclaimed = sum(r['size'] for r in plan['old_parquet'] if local_target(source,r['path']).is_file()) if delete_old else 0
            same_disk = source.stat().st_dev == root.stat().st_dev
            if (shutil.disk_usage(source).free+reclaimed < missing_bytes+headroom+(pcm_remaining if same_disk else 0)
                    or (not same_disk and shutil.disk_usage(root).free < pcm_remaining+headroom)):
                raise OSError('Insufficient space for Parquet plus conservative PCM estimate; no old Parquet deleted')
            if delete_old:
                cleanup = remove_old(plan)
                write_json(root/'cleanup_report.json',cleanup)
                log(f'[yo-expansion] Removed {cleanup["deleted_files"]} verified old Parquet files; WAVs/caches retained.')
            status.update(phase='download'); write_json(marker,status)
            for i,entry in enumerate(plan['selected'],1):
                path = local_target(source,entry['path'])
                if not path.is_file() and shutil.disk_usage(source).free < entry['size']+2*1024**3:
                    raise OSError('Insufficient free space for next Parquet; previous downloads are resumable')
                download_files(source,[entry['path']],REPO,plan['revision'],authenticated=True,
                               transport=transport,interval=2.,log=log)
                check_shard(source,entry)
                status.update(downloaded_shards=i,total_shards=len(plan['selected']))
                write_json(marker,status)
            status.update(phase='extract'); write_json(marker,status)
            parts = root/'parts'; parts.mkdir(exist_ok=True)
            state = dict(seen=set(),seconds=Counter(),total=0.)
            excluded = raw_utterances(groups['train']+groups['validation']+groups['protected'])
            all_rows, rejected = [], Counter()
            for i,entry in enumerate(plan['selected'],1):
                folder = parts/Path(entry['path']).stem
                if not folder.exists():
                    if shutil.disk_usage(root).free < entry['size']*2+2*1024**3:
                        raise OSError('Insufficient free space for next extracted shard; completed parts are retained')
                    before = set(state['seen'])
                    with tempfile.TemporaryDirectory(prefix='.extract-',dir=parts) as temporary:
                        staging = Path(temporary)/'data'
                        import_yo([source/entry['path']],staging,source_revision=plan['revision'],
                                  import_state=state,finalize_pairs=False,allow_empty=True,
                                  excluded_utterances=excluded,log=log,**plan['filters'])
                        write_json(staging/'part.json',dict(plan_sha256=fingerprint,shard=entry,
                            seen=sorted(state['seen']-before),manifest_sha256=file_digest(staging/'manifest.jsonl')))
                        staging.rename(folder)
                else:
                    receipt = json.loads((folder/'part.json').read_text())
                    if (receipt['plan_sha256'] != fingerprint or receipt['shard'] != entry
                            or receipt['manifest_sha256'] != file_digest(folder/'manifest.jsonl')):
                        raise ValueError('Extracted part changed; refusing to resume with different data')
                    state['seen'].update(receipt['seen'])
                    for row in checked_rows(folder/'manifest.jsonl'):
                        seconds = row['end']-row['start']
                        state['seconds'][row['speaker_id'].removeprefix('yo:')] += seconds
                        state['total'] += seconds
                rows = checked_rows(folder/'manifest.jsonl')
                all_rows.extend(rows)
                rejected.update(json.loads((folder/'import_report.json').read_text())['rejections'])
                status.update(extracted_shards=i,accepted_before_final_pairing=len(all_rows))
                write_json(marker,status)
                log(f'[yo-expansion] Extracted {i}/{len(plan["selected"])} shards; candidate clips={len(all_rows)}')
            known_audio = {r['audio_sha256'] for rows in groups.values() for r in rows}
            unique, hashes = [], set()
            for row in all_rows:
                if row['audio_sha256'] in known_audio:
                    rejected['previous_audio_duplicate'] += 1
                elif row['audio_sha256'] in hashes:
                    rejected['new_audio_duplicate'] += 1
                else:
                    unique.append(row); hashes.add(row['audio_sha256'])
            pairs = defaultdict(list)
            for row in unique:
                pairs[(row['speaker_id'],row['language'])].append(row)
            kept = [row for clips in pairs.values() for row in clips
                    if any(SpeakerPairs.distinct(row,ref) for ref in clips)]
            rejected['segments_without_distinct_reference'] += len(unique)-len(kept)
            train, validation = split_with_anchors(kept,groups['train'],groups['validation'],seed=plan['seed'])
            SpeakerPairs(train,None,None); SpeakerPairs(validation,None,None)
            write_manifest(root/'manifest.jsonl',kept)
            write_manifest(root/'train.jsonl',train); write_manifest(root/'validation.jsonl',validation)
            report = dict(plan_sha256=fingerprint,source_revision=plan['revision'],shards=len(plan['selected']),
                          rejections=dict(rejected),filters=plan['filters'],scope='speaker',
                          train_clips=len(train),validation_clips=len(validation),
                          train_hours=sum(r['duration'] for r in train)/3600,
                          validation_hours=sum(r['duration'] for r in validation)/3600,
                          train_speakers=len({r['speaker_id'] for r in train}),
                          validation_speakers=len({r['speaker_id'] for r in validation}),
                          longest_clip_seconds=max(r['duration'] for r in kept))
            kept_paths = {r['audio'] for r in kept}
            unused = [dict(path=Path(r['audio']).relative_to(root).as_posix(),sha256=r['audio_sha256'])
                      for r in all_rows if r['audio'] not in kept_paths]
            report['unused_extracted_audio'] = len(unused)
            write_json(root/'unused_audio.json',unused)
            register_used(source,plan['revision'],plan['selected'])
            write_json(root/'split_report.json',report)
            prune_unused(root,kept)
            status.update(status='completed',phase='complete',complete=True)
            write_json(marker,status)
            log('[yo-expansion] Complete. Review split_report.json before configuring GPU training.')
            return report
        except BaseException as error:
            status.update(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',error_type=type(error).__name__)
            write_json(marker,status)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command',required=True)
    plan = commands.add_parser('plan')
    plan.add_argument('--source',default='training/raw/yo')
    plan.add_argument('--output',default='training/plans/yo-half')
    plan.add_argument('--fraction',type=float,default=.5)
    plan.add_argument('--seed',type=int,default=1234)
    plan.add_argument('--revision',default=REVISION)
    plan.add_argument('--train-anchors',nargs='+',default=TRAIN)
    plan.add_argument('--validation-anchors',nargs='+',default=VALIDATION)
    plan.add_argument('--protected-manifests',nargs='*',default=PROTECTED)
    run = commands.add_parser('run')
    run.add_argument('plan')
    run.add_argument('--output',default='training/datasets/yo-half')
    run.add_argument('--delete-old-parquet',action='store_true',help='Delete only the old files recorded in the plan after checking historical WAVs')
    run.add_argument('--transport',choices=['xet','http'],default='xet')
    args = parser.parse_args()
    if args.command == 'plan':
        result = plan_expansion(args.source,args.output,args.fraction,args.seed,args.revision,
                                args.train_anchors,args.validation_anchors,args.protected_manifests)
    else:
        result = run_expansion(args.plan,args.output,args.delete_old_parquet,args.transport)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()

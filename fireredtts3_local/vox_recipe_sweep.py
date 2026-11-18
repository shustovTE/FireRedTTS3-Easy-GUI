"""Replay the levelled FireRed LoRA from update 30, saving adapters and matched audio every 100 updates."""
import argparse
import json
from pathlib import Path

from filelock import FileLock

from . import vox_recipe_levels
from .training_data import file_digest
from .training_job import write_json
from .training_monitor import SETTINGS, load_benchmark
from .vox_recipe_compare import compare
from .vox_recipe_transfer import BASE_SHA, OUTPUT, train


STEPS = (100, 200, 300, 400)


def sweep(dataset, source, reports):
    dataset, source = dataset.resolve(), source.resolve()
    pilot = source / 'lora' / 'step-0030'
    saved = json.loads((pilot / 'checkpoint.json').read_text())
    recipe = saved['recipe']
    if saved['step'] != 30 or recipe['mode'] != 'lora' or recipe['base_sha256'] != BASE_SHA:
        raise ValueError(f'Expected the original LoRA pilot at update 30: {pilot}')

    reports = [p.resolve() for p in reports]
    if not reports or len(set(reports)) != len(reports):
        raise ValueError('Supply distinct completed pilot reports')
    for p in reports:
        old = json.loads(p.read_text())
        _, fingerprint = load_benchmark(p.parent / 'benchmark.jsonl')
        if (not old['complete'] or old['step'] != 30 or old['recipes'].get('lora') != recipe
                or old['benchmark_sha256'] != fingerprint or old['settings'] != SETTINGS):
            raise ValueError(f'Report, reference or settings differ from the step-30 pilot: {p}')

    # Bind the separate replay to the actual adapter, optimizer/scheduler/RNG state and listening cases.
    plan = dict(pilot=str(pilot), files={name: file_digest(pilot / name) for name in
                ('checkpoint.json', 'adapter_config.json', 'adapter_model.safetensors', 'training_state.pt')},
                level_report_sha256=file_digest(dataset / vox_recipe_levels.REPORT),
                steps=list(STEPS), reports=[{'path': str(p), 'sha256': file_digest(p)} for p in reports])
    run = source / 'lora-0030-to-0400'
    run.mkdir(exist_ok=True)
    with FileLock(str(run / '.run.lock'), timeout=0):
        manifest = run / 'replay.json'
        if manifest.exists():
            if json.loads(manifest.read_text()) != plan:
                raise ValueError(f'Replay inputs differ from {manifest}')
        else:
            if (run / 'lora').exists() or (run / 'audio').exists():
                raise ValueError(f'Existing sweep outputs have no replay metadata: {run}')
            write_json(manifest, plan)
        for step in STEPS:
            checkpoint = run / 'lora' / f'step-{step:04d}'
            checkpoint_info = checkpoint / 'checkpoint.json'
            if not checkpoint_info.is_file():
                print(f'[sweep] Training LoRA to update {step}', flush=True)
                train(dataset, run, 'lora', step, True, fork_from=pilot if step == STEPS[0] else None,
                      shared_features=source / 'features')
            record = json.loads(checkpoint_info.read_text())
            if record['step'] != step or record['recipe'] != recipe:
                raise ValueError(f'Unexpected replay checkpoint: {checkpoint}')
            print(f'[sweep] Adapter: {checkpoint}', flush=True)
            for i, p in enumerate(reports, 1):
                folder = run / 'audio' / checkpoint.name / f'series-{i}'
                if not folder.exists():
                    old = json.loads(p.read_text())
                    path = compare(dataset, run, step, None, old['asr'], old['speaker_model'],
                                   against=p, lora_only=True)
                    folder.parent.mkdir(parents=True, exist_ok=True)
                    path.parent.rename(folder)
                print(f'[sweep] Listening WAVs: {folder / "lora"}', flush=True)
                print(f'[sweep] Report: {folder / "report.json"}', flush=True)
    return run


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reports', type=Path, nargs='+', help='Completed levelled step-30 comparison reports')
    parser.add_argument('--source', type=Path, default=Path(OUTPUT + '-levels'))
    parser.add_argument('--dataset', type=Path, default=Path(vox_recipe_levels.OUTPUT))
    args = parser.parse_args()
    print(sweep(args.dataset, args.source, args.reports))

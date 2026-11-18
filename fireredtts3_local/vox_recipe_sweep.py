"""Train and evaluate FireRed LoRA checkpoints on the levelled Russian data."""
import argparse
import json
from pathlib import Path

from filelock import FileLock

from . import vox_recipe_levels, vox_recovery
from .training_data import file_digest
from .training_job import write_json
from .training_monitor import SETTINGS, load_benchmark
from .vox_recipe_compare import compare, compare_recovery_strength, load_saved_base
from .vox_recipe_transfer import BASE_SHA, OUTPUT, paired_data, train, transfer_recipe


STEPS = (100, 200, 300, 400)


def fresh_recovery(dataset, source, reports, *, stronger_comb=False):
    dataset, source = dataset.resolve(), source.resolve()
    config = vox_recovery.STRONGER_COMB if stronger_comb else vox_recovery.CONFIG
    print(f"[sweep] Recovery from original Base, updates 1–100, comb_mix={config['comb_mix']:.2f}. "
          "Checking saved Base results and training data.", flush=True)
    reports = [p.resolve() for p in reports]
    if not reports or len(set(reports)) != len(reports):
        raise ValueError('Supply distinct reports with saved Base results')
    for p in reports:
        load_saved_base(p)
    _, _, selection = paired_data(dataset)
    recipe = dict(transfer_recipe('lora', selection['selection_sha256']),
                  acoustic_recovery=config, recovery_start_step=0)
    steps = [50, 100]
    plan = dict(initialization='fresh LoRA on original Base', recipe=recipe, steps=steps,
                level_report_sha256=file_digest(dataset / vox_recipe_levels.REPORT),
                reports=[{'path': str(p), 'sha256': file_digest(p)} for p in reports])
    standard = source / 'lora-base-recovery-0100'
    run = source / 'lora-base-recovery-comb050-0100' if stronger_comb else standard
    return run_slices(dataset, source, run, plan, recipe, {step: reports for step in steps},
                      reuse_base=True, standard_run=standard if stronger_comb else None)


def sweep(dataset, source, reports, *, until=STEPS[-1], control=None):
    if not 30 < until <= STEPS[-1]:
        raise ValueError(f'--until must be an absolute update between 31 and {STEPS[-1]}')
    steps = [step for step in STEPS if step < until] + [until]
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
                steps=steps, reports=[{'path': str(p), 'sha256': file_digest(p)} for p in reports])
    run = source / f'lora-0030-to-{until:04d}'
    comparisons = {step: reports for step in steps}
    if control is not None:
        if control != run:
            raise ValueError(f'Expected the matched control in {run}')
        recipe = dict(recipe, acoustic_recovery=vox_recovery.CONFIG)
        run = run.with_name(run.name + '-recovery')
        plan['acoustic_recovery'] = vox_recovery.CONFIG
        plan['controls'] = {}
        for step in steps:
            comparisons[step] = []
            for i, p in enumerate(reports, 1):
                report = control / 'audio' / f'step-{step:04d}' / f'series-{i}' / 'report.json'
                baseline = json.loads(report.read_text())
                pilot_report = json.loads(p.read_text())
                if (not baseline['complete'] or baseline['step'] != step or baseline['compared_to'] != str(p)
                        or baseline['benchmark_sha256'] != pilot_report['benchmark_sha256']
                        or baseline['recipes']['lora'] != saved['recipe']):
                    raise ValueError(f'Mismatched clean control: {report}')
                plan['controls'][str(report)] = file_digest(report)
                comparisons[step].append(report)
    return run_slices(dataset, source, run, plan, recipe, comparisons, pilot=pilot)


def run_slices(dataset, source, run, plan, recipe, comparisons, *, pilot=None, reuse_base=False, standard_run=None):
    run.mkdir(parents=True, exist_ok=True)
    with FileLock(str(run / '.run.lock'), timeout=0):
        manifest = run / 'replay.json'
        if manifest.exists():
            if json.loads(manifest.read_text()) != plan:
                raise ValueError(f'Replay inputs differ from {manifest}')
        else:
            if (run / 'lora').exists() or (run / 'audio').exists():
                raise ValueError(f'Existing sweep outputs have no replay metadata: {run}')
            write_json(manifest, plan)
        for index, (step, reports) in enumerate(comparisons.items()):
            checkpoint = run / 'lora' / f'step-{step:04d}'
            checkpoint_info = checkpoint / 'checkpoint.json'
            if not checkpoint_info.is_file():
                print(f'[sweep] Training LoRA to update {step}', flush=True)
                train(dataset, run, 'lora', step, resume=index > 0 or pilot is not None,
                      fork_from=pilot if index == 0 else None,
                      shared_features=source / 'features', recovery=recipe.get('acoustic_recovery', False))
            record = json.loads(checkpoint_info.read_text())
            if record['step'] != step or record['recipe'] != recipe:
                raise ValueError(f'Unexpected replay checkpoint: {checkpoint}')
            print(f'[sweep] Adapter: {checkpoint}', flush=True)
            for i, p in enumerate(reports, 1):
                folder = run / 'audio' / checkpoint.name / f'series-{i}'
                if not folder.exists():
                    old = json.loads(p.read_text())
                    path = compare(dataset, run, step, None, old['asr'], old['speaker_model'],
                                   against=p, lora_only=True, reuse_base=reuse_base)
                    folder.parent.mkdir(parents=True, exist_ok=True)
                    path.parent.rename(folder)
                print(f'[sweep] Listening WAVs: {folder / "lora"}', flush=True)
                print(f'[sweep] Report: {folder / "report.json"}', flush=True)
                if standard_run is not None:
                    standard = standard_run / folder.relative_to(run) / 'report.json'
                    try:
                        paired = compare_recovery_strength(standard, folder / 'report.json')
                    except (OSError, KeyError, TypeError, ValueError) as exc:
                        # A missing or incompatible old report should only affect the optional A/B comparison.
                        paired = dict(complete=False, step=step, standard_report=str(standard), reason=str(exc), cases=[])
                        print(f'[sweep] Could not pair these reports: {standard}: {exc}', flush=True)
                    write_json(folder / 'paired-recovery.json', paired)
                    print(f"[sweep] Paired recovery comparison: {folder / 'paired-recovery.json'}", flush=True)
    return run


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reports', type=Path, nargs='+', help='Existing comparison reports for the listening cases')
    parser.add_argument('--source', type=Path, default=Path(OUTPUT + '-levels'))
    parser.add_argument('--dataset', type=Path, default=Path(vox_recipe_levels.OUTPUT))
    parser.add_argument('--until', type=int,
                        help='Absolute last update (31–400). Save and evaluate there and at earlier hundreds.')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--recovery', action='store_true',
                      help='Also train from the same pilot with damaged acoustic histories and compare to the clean run.')
    mode.add_argument('--recovery-from-base', action='store_true',
                      help='Train recovery LoRA from Base to 100. Save adapters and only adapted audio at 50 and 100.')
    parser.add_argument('--stronger-comb', action='store_true',
                        help='With --recovery-from-base, raise only the delayed signal coefficient from 0.35 to 0.50.')
    args = parser.parse_args()
    if args.stronger_comb and not args.recovery_from_base:
        parser.error('--stronger-comb requires --recovery-from-base')
    if args.recovery_from_base:
        if args.until not in (None, 100):
            parser.error('--recovery-from-base ends at update 100')
        print(fresh_recovery(args.dataset, args.source, args.reports, stronger_comb=args.stronger_comb))
    else:
        until = STEPS[-1] if args.until is None else args.until
        control = sweep(args.dataset, args.source, args.reports, until=until)
        print(control, flush=True)
        if args.recovery:
            print(sweep(args.dataset, args.source, args.reports, until=until, control=control))

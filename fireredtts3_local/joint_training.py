"""One fresh-base hybrid run on the existing original Dialogs and confirmed CV data."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import tempfile

from filelock import FileLock

from .common_voice_spontaneous import SCHEMA as CV_SCHEMA, validate_selection
from .dialogs_dataset import lexical_key
from .dialogs_only import audit_prepared as audit_dialogs
from .training_data import dataset_fingerprint, file_digest, write_manifest
from .training_monitor import load_benchmark
from .training_run import TrainingOptions, prepare_run_data

SCHEMA = 'russian-joint-hybrid-v1'
DEFAULT_OUTPUT = 'training/datasets/russian-joint-10'
DEFAULT_RECIPE = 'recipes/russian_joint_96gb.json'


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def describe(rows):
    return dict(clips=len(rows), hours=sum(r['duration'] for r in rows)/3600,
                speakers=len({r['speaker_id'] for r in rows}), fingerprint=dataset_fingerprint(rows))


def validate_recipe(config):
    opts = TrainingOptions.parse(config['options'])
    if (config.get('initial_adapter') or config.get('resume_checkpoint')
            or config['length_mode'] != 'Epochs' or opts.profile != 'hybrid'
            or opts.sampling_strategy != 'coverage' or opts.replay_manifest or opts.replay_ratio
            or opts.validation_scope != 'recording' or not opts.validation_all_targets
            or opts.max_target_seconds != 30 or opts.max_prompt_seconds != 30
            or opts.precision_policy != 'native' or not opts.expected_base_fingerprint
            or not opts.generation_benchmark):
        raise ValueError('Require fresh pinned base, hybrid/native, union coverage, both holdouts and generation benchmark')
    return opts


def audit(output=DEFAULT_OUTPUT):
    output = Path(output).resolve()
    report = json.loads((output/'report.json').read_text())
    if report.get('schema') != SCHEMA or not report.get('complete'):
        raise ValueError('Complete joint preparation first')
    if file_digest(output/'recipe.json') != report['recipe_sha256']:
        raise ValueError('Prepared joint recipe changed')
    config = json.loads((output/'recipe.json').read_text())
    opts = validate_recipe(config)
    train, validation, replay, adaptation = prepare_run_data(config['manifest_path'], opts, config['seed'],
                                                            include_adaptation_validation=True)
    if replay or adaptation is None:
        raise ValueError('Expected one combined train and two separate holdouts')
    for name, pairs in [('train',train),('dialogs_validation',validation),('spontaneous_validation',adaptation)]:
        if dataset_fingerprint(pairs.rows) != report['datasets'][name]['fingerprint']:
            raise ValueError(f'Joint {name} changed')
    heldout = validation.rows + adaptation.rows
    if {lexical_key(r['text']) for r in train.rows} & {lexical_key(r['text']) for r in heldout}:
        raise ValueError('Joint holdout text leakage')
    _, signature = load_benchmark(opts.generation_benchmark)
    if signature != report['generation_benchmark_fingerprint']:
        raise ValueError('Generation benchmark/reference changed')
    return report


def prepare(dialogs='training/datasets/dialogs-base-08',
            spontaneous='training/datasets/cv-spontaneous-09', output=DEFAULT_OUTPUT,
            recipe=DEFAULT_RECIPE):
    dialogs, spontaneous, output = map(lambda p:Path(p).resolve(), (dialogs,spontaneous,output))
    config = json.loads(Path(recipe).read_text())
    opts = validate_recipe(config)
    run = Path(config['output_dir']).resolve()
    if output.exists() or (run.exists() and any(run.iterdir())):
        raise ValueError('Joint preparation/run must be new; use audit or resume existing work')
    for a,b in ((output,dialogs),(output,spontaneous),(output,run),(run,dialogs),(run,spontaneous)):
        if a == b or a.is_relative_to(b) or b.is_relative_to(a):
            raise ValueError('Joint output/run must be separate from each other and source data')
    dialogs_report, dialogs_config, _ = audit_dialogs(dialogs)
    dt,dv,_ = prepare_run_data(dialogs_config['manifest_path'],dialogs_config['options'],dialogs_config['seed'])
    cv_report = json.loads((spontaneous/'report.json').read_text())
    if cv_report.get('schema') != CV_SCHEMA or not cv_report.get('complete'):
        raise ValueError('Require the existing confirmed Common Voice preparation')
    if file_digest(spontaneous/'recipe.json') != cv_report['recipe_sha256']:
        raise ValueError('Source Common Voice recipe changed')
    cv_config = json.loads((spontaneous/'recipe.json').read_text())
    ct,cv,old_dt,old_dv = prepare_run_data(cv_config['manifest_path'],cv_config['options'],cv_config['seed'],
                                         include_adaptation_validation=True)
    for name,pairs in [('train',ct),('validation',cv),('dialogs_train',old_dt),('dialogs_validation',old_dv)]:
        if pairs is None or dataset_fingerprint(pairs.rows) != cv_report['datasets'][name]['fingerprint']:
            raise ValueError(f'Source {name} audio/metadata changed')
    for split,pairs in [('train',old_dt),('validation',old_dv)]:
        if dataset_fingerprint(pairs.rows) != dialogs_report['datasets'][split]['fingerprint']:
            raise ValueError('Common Voice and Dialogs refer to different original selections')
    validate_selection(ct.rows,cv.rows)
    # The old stage's adapter is deliberately not read or required.
    for source_config in (dialogs_config,cv_config):
        if source_config['options']['expected_base_fingerprint'] != opts.expected_base_fingerprint:
            raise ValueError('Source/base provenance mismatch')
    cases,signature = load_benchmark(opts.generation_benchmark)
    train = dt.rows + ct.rows
    if {lexical_key(r['text']) for r in train} & {lexical_key(r['text']) for r in dv.rows+cv.rows}:
        raise ValueError('Cross-corpus train/validation text leakage')
    if {r['audio_sha256'] for r in train} & {r['reference_sha256'] for r in cases}:
        raise ValueError('Generation benchmark voice reference occurs in training')
    if {lexical_key(r['text']) for r in train} & {lexical_key(r['text']) for r in cases}:
        raise ValueError('Generation benchmark text occurs in training')
    output.parent.mkdir(parents=True,exist_ok=True)
    with FileLock(str(output)+'.prepare.lock',timeout=0):
        if output.exists():
            raise ValueError('Joint preparation already exists')
        with tempfile.TemporaryDirectory(prefix='.joint-',dir=output.parent) as temporary:
            staging = Path(temporary)/'prepared';staging.mkdir()
            for name,rows in [('train',train),('dialogs-validation',dv.rows),('spontaneous-validation',cv.rows)]:
                write_manifest(staging/f'{name}.jsonl',rows)
            write_manifest(staging/'benchmark.jsonl',cases)
            # Validate the actual training loader before atomically installing.
            checked_opts = {**config['options'], 'validation_manifest':str(staging/'dialogs-validation.jsonl'),
                            'adaptation_validation_manifest':str(staging/'spontaneous-validation.jsonl')}
            checked = prepare_run_data(staging/'train.jsonl',checked_opts,config['seed'],include_adaptation_validation=True)
            config.update(manifest_path=str(output/'train.jsonl'),output_dir=str(run))
            config['options'].update(validation_manifest=str(output/'dialogs-validation.jsonl'),
                adaptation_validation_manifest=str(output/'spontaneous-validation.jsonl'),
                generation_benchmark=str(output/'benchmark.jsonl'),cache_dir=str(Path(opts.cache_dir).resolve()))
            write_json(staging/'recipe.json',config)
            steps=math.ceil(len(train)/(opts.batch_size*config['gradient_accumulation']))
            report=dict(schema=SCHEMA,complete=True,initialization='original base; fresh hybrid; no previous adapter',
                sources=dict(dialogs=str(dialogs),spontaneous=str(spontaneous)),
                source_reports=dict(dialogs=file_digest(dialogs/'report.json'),spontaneous=file_digest(spontaneous/'report.json')),
                datasets={name:describe(pairs.rows) for name,pairs in
                    [('train',checked[0]),('dialogs_validation',checked[1]),('spontaneous_validation',checked[3])]},
                training_corpora=dict(dialogs=describe(dt.rows),spontaneous=describe(ct.rows)),
                updates_per_epoch=steps,maximum_steps=steps*config['training_epochs'],
                audio='existing full recordings, no new cropping or normalization',
                sampling='each combined target once per epoch; no replay or oversampling',
                generation_cases=len(cases),generation_benchmark_fingerprint=signature,
                recipe=str(output/'recipe.json'),recipe_sha256=file_digest(staging/'recipe.json'))
            write_json(staging/'report.json',report)
            staging.rename(output)
    return audit(output)


def resume_recipe(job, output):
    from .training_job import read_status
    job=Path(job).resolve();state=read_status(job)
    if state['active'] or state['status'] not in {'completed','stopped'}:
        raise ValueError('Wait for the pilot to finish before creating a resume recipe')
    config=json.loads((job/'recipe.json').read_text())
    if config['options']['profile'] != 'hybrid' or config.get('initial_adapter'):
        raise ValueError('Expected a fresh-base hybrid job')
    run=Path(config['output_dir']).resolve()
    if not (run/'trainer_state.pt').is_file():
        raise ValueError('Pilot has no saved trainer state')
    config['resume_checkpoint']=str(run)
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as stream:
        stream.write(json.dumps(config,ensure_ascii=False,indent=2)+'\n')
    return {'resume_recipe':str(output.resolve())}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    p=commands.add_parser('prepare');p.add_argument('--dialogs',default='training/datasets/dialogs-base-08')
    p.add_argument('--spontaneous',default='training/datasets/cv-spontaneous-09')
    p.add_argument('--output',default=DEFAULT_OUTPUT);p.add_argument('--recipe',default=DEFAULT_RECIPE)
    p=commands.add_parser('audit');p.add_argument('output',nargs='?',default=DEFAULT_OUTPUT)
    p=commands.add_parser('resume');p.add_argument('--job',required=True);p.add_argument('--output',required=True)
    args=vars(parser.parse_args());command=args.pop('command')
    result={'prepare':prepare,'audit':audit,'resume':resume_recipe}[command](**args)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()

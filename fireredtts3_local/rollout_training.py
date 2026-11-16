"""Prepare/audit/resume one fresh-base, own-history Dialogs + Spontaneous run."""
import argparse
import json
from pathlib import Path

from . import joint_training

DEFAULT_OUTPUT = 'training/datasets/russian-rollout-11'
DEFAULT_RECIPE = 'recipes/russian_rollout_96gb.json'
DEFAULT_RUN = 'training/outputs/russian-rollout-base-11'
CHECK_OUTPUT = 'training/eval/rollout-11-check'
CHECK_JOB = 'training/jobs/rollout-11-check'


def prepare(dialogs='training/datasets/dialogs-base-08',
            spontaneous='training/datasets/cv-spontaneous-09',
            output=DEFAULT_OUTPUT, recipe=DEFAULT_RECIPE):
    return joint_training.prepare(dialogs, spontaneous, output, recipe, mode='rollout')


def audit(output=DEFAULT_OUTPUT):
    return joint_training.audit(output, mode='rollout')


def resume_recipe(job, output):
    return joint_training.resume_recipe(job, output, mode='rollout')


def check(audio, run=DEFAULT_RUN, output=CHECK_OUTPUT, job_dir=CHECK_JOB,
          steps=(143, 240), resume=False):
    """Resolve an existing monitor WAV to its exact text/reference/seed case.

    Three unsplit generations by default: original base, epoch-one checkpoint,
    and the pilot's last checkpoint. Shared launcher locks the source run and
    verifies adapter/base/benchmark fingerprints; no training state is written.
    """
    from .cfg_diagnostics import launch
    run = Path(run).resolve()
    recipe = json.loads((run/'recipe.json').read_text())
    if recipe['options']['profile'] != 'backbone' or recipe['options'].get('rollout') is None:
        raise ValueError('Expected the own-history backbone run')
    baseline = json.loads((run/'generation_baseline.json').read_text())
    if (not baseline.get('complete') or baseline.get('step') != 0
            or baseline.get('fingerprint') != recipe['generation_benchmark_fingerprint']):
        raise ValueError('Original generation baseline does not match the training run')
    matches = [r for r in baseline['results'] if r.get('audio') == audio]
    if len(matches) != 1:
        raise ValueError('Specify one exact monitor WAV name, e.g. case-00006.wav')
    case_id = matches[0]['id']
    result = launch(run, output, job_dir, steps=steps, cfgs=(2.,), cases=[case_id], resume=resume)
    return {**result, 'case_id': case_id, 'source_audio': audio, 'training_updates': 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--dialogs', default='training/datasets/dialogs-base-08')
    p.add_argument('--spontaneous', default='training/datasets/cv-spontaneous-09')
    p.add_argument('--output', default=DEFAULT_OUTPUT)
    p.add_argument('--recipe', default=DEFAULT_RECIPE)
    p = sub.add_parser('audit'); p.add_argument('output', nargs='?', default=DEFAULT_OUTPUT)
    p = sub.add_parser('resume'); p.add_argument('--job', required=True); p.add_argument('--output', required=True)
    p = sub.add_parser('check', help='Compare base and saved checkpoints without training; whole utterance only')
    p.add_argument('--audio', required=True, help='Exact WAV name in the training generation report')
    p.add_argument('--run', default=DEFAULT_RUN); p.add_argument('--output', default=CHECK_OUTPUT)
    p.add_argument('--job-dir', default=CHECK_JOB); p.add_argument('--steps', type=int, nargs='+', default=[143,240])
    p.add_argument('--resume', action='store_true')
    args = vars(parser.parse_args()); command = args.pop('command')
    print(json.dumps({'prepare': prepare, 'audit': audit, 'resume': resume_recipe, 'check': check}[command](**args),
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

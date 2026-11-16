"""Prepare/audit/resume one fresh-base, own-history Dialogs + Spontaneous run."""
import argparse
import json

from . import joint_training

DEFAULT_OUTPUT = 'training/datasets/russian-rollout-11'
DEFAULT_RECIPE = 'recipes/russian_rollout_96gb.json'


def prepare(dialogs='training/datasets/dialogs-base-08',
            spontaneous='training/datasets/cv-spontaneous-09',
            output=DEFAULT_OUTPUT, recipe=DEFAULT_RECIPE):
    return joint_training.prepare(dialogs, spontaneous, output, recipe, mode='rollout')


def audit(output=DEFAULT_OUTPUT):
    return joint_training.audit(output, mode='rollout')


def resume_recipe(job, output):
    return joint_training.resume_recipe(job, output, mode='rollout')


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
    args = vars(parser.parse_args()); command = args.pop('command')
    print(json.dumps({'prepare': prepare, 'audit': audit, 'resume': resume_recipe}[command](**args),
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

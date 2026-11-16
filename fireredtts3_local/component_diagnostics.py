"""Native checkpoint ablation: backbone LoRA, dense acoustics, and stop head."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import cfg_diagnostics as shared

DEFAULT_OUTPUT = 'training/eval/joint-10-component-diagnostics'
DEFAULT_JOB = 'training/jobs/joint-10-components'
SCHEMA = 'native-component-diagnostics-v1'
SPLIT_SCHEMA = 'native-acoustic-components-v1'
SPLIT_OUTPUT = 'training/eval/joint-10-acoustic-diagnostics'
SPLIT_JOB = 'training/jobs/joint-10-acoustic'
COMPONENTS = ('backbone', 'acoustic', 'stop')
DEFINITIONS = {
    'backbone': 'Trained backbone_llm LoRA; disabled uses the original backbone weights.',
    'acoustic': 'Trained full dit and dit_head; disabled uses both original base modules.',
    'stop': 'Trained full stop_head; disabled uses the original base stop head.',
}
VARIANTS = (
    ('base', ()),
    ('all', COMPONENTS),
    ('backbone_only', ('backbone',)),
    ('acoustic_only', ('acoustic',)),
    ('stop_only', ('stop',)),
    ('backbone_acoustic', ('backbone', 'acoustic')),
    ('backbone_stop', ('backbone', 'stop')),
    ('acoustic_stop', ('acoustic', 'stop')),
)
SPLIT_DEFINITIONS = {
    'backbone': DEFINITIONS['backbone'],
    'dit': 'Trained full DiT; disabled uses the original base DiT.',
    'projection': 'Trained dit_head conditioning projection; disabled uses the original dit_head.',
    'stop': 'Trained stop_head only in the all-components reproduction control; original in the eight-way matrix.',
}
SPLIT_VARIANTS = (
    ('base', ()),
    ('all', ('backbone', 'dit', 'projection', 'stop')),
    ('backbone_dit_projection', ('backbone', 'dit', 'projection')),
    ('backbone_only', ('backbone',)),
    ('dit_only', ('dit',)),
    ('projection_only', ('projection',)),
    ('backbone_dit', ('backbone', 'dit')),
    ('backbone_projection', ('backbone', 'projection')),
    ('dit_projection', ('dit', 'projection')),
)


def make_plan(run=shared.DEFAULT_RUN, output=DEFAULT_OUTPUT, step=143, cases=shared.DEFAULT_CASES,
              *, split_acoustic=False):
    plan = shared.make_plan(run, output, steps=(step,), cfgs=(2.,), cases=cases)
    if plan['adapter_profile'] != 'hybrid':
        raise ValueError('Component ablation requires a hybrid adapter with trained dense modules')
    if str(step) not in plan['previous_reports']:
        raise ValueError('The selected checkpoint needs a previous generation report for the all-components control')
    # Both controls must have every requested case; a missing comparison cannot
    # masquerade as successful reproduction.
    for control in ('0', str(step)):
        report = json.loads(Path(plan['previous_reports'][control]['path']).read_text())
        ids = [r['id'] for r in report['results']]
        if len(ids) != len(set(ids)) or not set(cases) <= set(ids):
            raise ValueError(f'Previous control report at step {control} has missing or duplicate cases')
    base, trained = plan['stages']
    variants = SPLIT_VARIANTS if split_acoustic else VARIANTS
    plan.update(schema=SPLIT_SCHEMA if split_acoustic else SCHEMA, ablation_step=step,
                component_definitions=SPLIT_DEFINITIONS if split_acoustic else DEFINITIONS,
                stages=[dict(base if name == 'base' else trained,
                             name=name, active_components=list(active),
                             comparison_step=0 if name == 'base' else step if name == 'all' else None)
                        for name, active in variants], total_cases=len(variants)*len(cases))
    if split_acoustic:
        plan['training_parity'] = True
    return plan


def select_components(core, active):
    """Switch PEFT routing only; never merge, zero or overwrite a parameter.

    A disabled ModulesToSaveWrapper routes through its frozen original_module.
    Validate the entire hybrid structure before switching any layers, then freeze
    again because PEFT enable_adapters also adjusts requires_grad flags.
    """
    from peft import PeftModel
    from peft.tuners.lora.layer import LoraLayer
    from peft.utils.other import ModulesToSaveWrapper

    order = ('backbone', 'acoustic', 'dit', 'projection', 'stop')
    if len(set(active)) != len(active) or set(active) - set(order):
        raise ValueError('Select unique components from backbone, acoustic, dit, projection, stop')
    if 'acoustic' in active and {'dit', 'projection'} & set(active):
        raise ValueError('Select acoustic as a group or dit/projection separately')
    active = [name for name in order if name in active]
    enabled_dense = {'dit': 'dit' in active or 'acoustic' in active,
                     'dit_head': 'projection' in active or 'acoustic' in active,
                     'stop_head': 'stop' in active}
    if not isinstance(core, PeftModel) or set(core.peft_config) != {'default'}:
        raise ValueError('Component diagnostics require one unmerged default hybrid adapter')
    base = core.get_base_model()
    lora, dense = {}, {}
    for name, layer in base.named_modules():
        if isinstance(layer, LoraLayer):
            if (not name.startswith('backbone_llm.') or layer.merged
                    or set(layer.lora_A) != {'default'} or set(layer.lora_B) != {'default'}
                    or list(layer.active_adapters) != ['default']):
                raise ValueError(f'Expected unmerged default backbone LoRA: {name}')
            lora[name] = layer
        elif isinstance(layer, ModulesToSaveWrapper):
            if (name not in ('dit', 'dit_head', 'stop_head')
                    or set(layer.modules_to_save) != {'default'}
                    or list(layer.active_adapters) != ['default']
                    or layer.original_module is None):
                raise ValueError(f'Unexpected dense adapter wrapper: {name}')
            dense[name] = layer
    if not lora or set(dense) != {'dit', 'dit_head', 'stop_head'}:
        raise ValueError('Require backbone LoRA and exactly dit, dit_head, stop_head dense wrappers')
    for layer in lora.values():
        layer.enable_adapters('backbone' in active)
    for name, layer in dense.items():
        layer.enable_adapters(enabled_dense[name])
    core.eval()
    core.requires_grad_(False)
    if (any(layer.disable_adapters != ('backbone' not in active) for layer in lora.values())
            or any(layer.disable_adapters == enabled_dense[name]
                   for name, layer in dense.items())):
        raise RuntimeError('PEFT did not apply the requested component routing')
    return dict(active_components=active, backbone_lora_layers=len(lora),
                enabled_backbone_lora_layers=len(lora) if 'backbone' in active else 0,
                dense_modules={name: 'trained' if not layer.disable_adapters else 'original_base'
                               for name, layer in dense.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    start = commands.add_parser('start', help='Run component variants in a detached job')
    start.add_argument('--run', default=shared.DEFAULT_RUN)
    start.add_argument('--output')
    start.add_argument('--job-dir')
    start.add_argument('--split-acoustic', action='store_true',
                       help='Separate DiT and dit_head: eight variants with base stop, plus full control; forward parity audit')
    start.add_argument('--resume', action='store_true')
    start.add_argument('--step', type=int, default=143)
    start.add_argument('--cases', nargs='+', default=shared.DEFAULT_CASES)
    for name in ('status', 'stop'):
        sub = commands.add_parser(name)
        sub.add_argument('job', nargs='?', default=DEFAULT_JOB)
    sub = commands.add_parser('summary')
    sub.add_argument('output', nargs='?', default=DEFAULT_OUTPUT)
    args = vars(parser.parse_args())
    command = args.pop('command')
    if command == 'start':
        step = args.pop('step')
        args['output'] = args['output'] or (SPLIT_OUTPUT if args['split_acoustic'] else DEFAULT_OUTPUT)
        args['job_dir'] = args['job_dir'] or (SPLIT_JOB if args['split_acoustic'] else DEFAULT_JOB)
        result = shared.launch(**args, component_step=step)
    elif command == 'summary':
        print((Path(args['output'])/'summary.md').read_text())
        return 0
    else:
        job = Path(args['job']).resolve()
        result = shared.read_status(job)
        if command == 'stop':
            if not result['active']:
                raise ValueError('Diagnostic job is not active')
            (job/'STOP').touch()
            result = dict(stop_requested=True, job_dir=str(job))
        else:
            report = Path(result['output_dir'])/'report.json'
            if report.is_file():
                result['completed_cases'] = json.loads(report.read_text())['completed_cases']
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

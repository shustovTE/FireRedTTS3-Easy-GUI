"""Train FireRed attention LoRA and render the listening set with backbone LoRA active."""
import argparse
import json
from importlib.metadata import version
from pathlib import Path

from filelock import FileLock

from . import vox_attention_ablation as listening
from . import redae_decoder_training, vox_recipe_levels, vox_recipe_transfer, vox_recovery
from .paths import ROOT
from .training_data import file_digest
from .training_job import write_json
from .training_monitor import load_benchmark
from .training_objective import BACKBONE_ATTENTION
from .vox_recipe_compare import load_saved_base


SOURCE = ROOT / (vox_recipe_transfer.OUTPUT + '-levels')
PILOTS = ('comparison-0030-yrslvj2s', 'comparison-0030-ql21g14n')
RUN = 'lora-backbone-only-recovery-4patch-0250'
CONTINUATION = 'lora-backbone-only-recovery-4patch-0500'
STRONG_RUN = 'lora-backbone-only-recovery-4patch-0500-resonant-w030'
JOINT_RUN = 'lora-joint-recovery-4patch-0500-backbone-audio'
DECODER_RUN = 'lora-backbone-redae-decoder-4patch-0500'


def check_step(checkpoint, step, recipe):
    record = json.loads((checkpoint / 'checkpoint.json').read_text())
    if record['step'] != step or record['recipe'] != recipe:
        raise ValueError(f'Unexpected training checkpoint at step {step}: {checkpoint}')


def run(source=SOURCE, dataset=ROOT / vox_recipe_levels.OUTPUT, *, continue_to_500=False,
        stronger_recovery_loss=False, joint_to_500=False, backbone_redae_to_500=False):
    if sum((continue_to_500, stronger_recovery_loss, joint_to_500, backbone_redae_to_500)) > 1:
        raise ValueError('Select one training mode')
    source, dataset = Path(source).resolve(), Path(dataset).resolve()
    run_name = CONTINUATION if continue_to_500 else RUN
    recovery = vox_recovery.FOUR_PATCH
    checkpoint_steps = [100, 250]
    if stronger_recovery_loss:
        run_name = STRONG_RUN
        recovery = vox_recovery.STRONGER_RESONANT
        checkpoint_steps = [100, 250, 500]
    both_slices = joint_to_500 or backbone_redae_to_500
    if both_slices:
        run_name = DECODER_RUN if backbone_redae_to_500 else JOINT_RUN
        checkpoint_steps = [250, 500]
    output = source / run_name
    output.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output / '.run.lock'), timeout=0):
        controls, datasets = [], set()
        for name in PILOTS:
            folder = source / name
            cases, digest = load_benchmark(folder / 'benchmark.jsonl')
            report, _ = load_saved_base(folder / 'report.json', benchmark=(cases, digest))
            datasets.add(report['recipes']['lora']['dataset_sha256'])
            controls.append(dict(series=str(folder), cases=cases, benchmark_sha256=digest))
        if len(datasets) != 1:
            raise ValueError('Both pilot series must use the same normalized training set')
        runtime = {pkg: version(pkg) for pkg in ('torch', 'torchaudio', 'peft', 'transformers')}
        plan = listening.backbone_listening_pack({}, *controls, runtime)
        plan['adapter_profile'] = 'vox_attention' if joint_to_500 else BACKBONE_ATTENTION
        dataset_sha = datasets.pop()
        recipe = dict(vox_recipe_transfer.transfer_recipe('lora', dataset_sha),
                      acoustic_recovery=recovery, recovery_start_step=0)
        # The original joint transfer recipe uses vox_attention by default in its checkpoint metadata.
        if not joint_to_500:
            recipe['adapter_profile'] = BACKBONE_ATTENTION
        if backbone_redae_to_500:
            plan['redae_decoder'] = recipe['redae_decoder'] = redae_decoder_training.CONFIG
        audio_steps = checkpoint_steps if both_slices else [checkpoint_steps[-1]]
        replay = dict(recipe=recipe, checkpoint_steps=checkpoint_steps, audio_steps=audio_steps, listening_plan=plan)
        parent = None
        if continue_to_500:
            parent = source / RUN / 'lora' / 'step-0250'
            check_step(parent, 250, recipe)
            if json.loads((source / RUN / 'replay.json').read_text()) != replay:
                raise ValueError('The training or listening inputs differ from the original 250-step run')
            parent_info = listening.checkpoint_files(parent)[parent.name]
            for name in ('training_state.pt', 'firered_lora_metadata.json'):
                parent_info['hashes'][name] = file_digest(parent / name)
            replay.update(parent_checkpoint=parent_info, checkpoint_steps=[500], audio_steps=[500])
        manifest = output / 'replay.json'
        if manifest.exists():
            if json.loads(manifest.read_text()) != replay:
                raise ValueError(f'Existing run has different training or listening inputs: {manifest}')
        else:
            if (output / 'lora').exists() or (output / 'audio').exists():
                raise ValueError(f'Existing outputs have no matching replay metadata: {output}')
            write_json(manifest, replay)

        print(f'[backbone] Train {plan["adapter_profile"]}, {recovery["schema"]}, recovery weight {recovery["weight"]:.2f}. '
              f'Weights at {replay["checkpoint_steps"]}; {len(plan["cases"])} backbone WAVs at each of '
              f'{replay["audio_steps"]}.', flush=True)
        for step in replay['audio_steps']:
            checkpoint = output / 'lora' / f'step-{step:04d}'
            if not both_slices or not (checkpoint / 'checkpoint.json').is_file():
                local_resume = any((output / 'lora').glob('step-*/checkpoint.json'))
                weights_only = tuple(s for s in replay['checkpoint_steps'] if s < step and s not in replay['audio_steps'])
                vox_recipe_transfer.train(dataset, output, 'lora', step, resume=local_resume or parent is not None,
                                          fork_from=None if local_resume else parent, shared_features=source / 'features',
                                          recovery=recovery, backbone_only=not joint_to_500, checkpoint_steps=weights_only,
                                          expected_dataset_sha256=dataset_sha,
                                          **({'redae_decoder': True} if backbone_redae_to_500 else {}))
            check_step(checkpoint, step, recipe)
            checkpoints = listening.checkpoint_files(checkpoint)
            if backbone_redae_to_500:
                redae_decoder_training.verify(checkpoint)
                checkpoints[checkpoint.name]['hashes'].update(redae_decoder_training.saved_hashes(checkpoint))
            render_plan = dict(plan, checkpoints=checkpoints)
            pack_name = 'backbone-redae-complete' if backbone_redae_to_500 else 'backbone-complete'
            pack = output / 'audio' / checkpoint.name / pack_name
            listening.render(pack, render_plan, audio_only=False)
            print(f'[backbone] Finished step {step}. Audio: {pack}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--continue-to-500', action='store_true',
                      help='Restore the completed backbone-only step 250 and train updates 251 through 500.')
    mode.add_argument('--stronger-recovery-loss', action='store_true',
                      help='Start from Base, train 500 steps with resonant history and recovery weight 0.30, '
                           'and render the listening WAVs only at step 500.')
    mode.add_argument('--joint-to-500', action='store_true',
                      help='Start from Base with backbone + DiT LoRA and standard four-patch recovery. '
                           'Render only the backbone part at steps 250 and 500.')
    mode.add_argument('--backbone-redae-to-500', action='store_true',
                      help='Train backbone and RedAE decoder attention adapters from Base. '
                           'Render both together at steps 250 and 500, using the original DiT and encoder.')
    run(**vars(parser.parse_args()))

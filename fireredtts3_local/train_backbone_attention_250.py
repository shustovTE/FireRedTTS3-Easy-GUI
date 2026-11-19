"""Train backbone Q/K/V/O LoRA to 250 or continue its saved state from 250 to 500."""
import argparse
import json
from importlib.metadata import version
from pathlib import Path

from filelock import FileLock

from . import vox_attention_ablation as listening
from . import vox_recipe_levels, vox_recipe_transfer, vox_recovery
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


def check_step(checkpoint, step, recipe):
    record = json.loads((checkpoint / 'checkpoint.json').read_text())
    if record['step'] != step or record['recipe'] != recipe:
        raise ValueError(f'Unexpected backbone-only checkpoint at step {step}: {checkpoint}')


def run(source=SOURCE, dataset=ROOT / vox_recipe_levels.OUTPUT, *, continue_to_500=False):
    source, dataset = Path(source).resolve(), Path(dataset).resolve()
    final_step = 500 if continue_to_500 else 250
    output = source / (CONTINUATION if continue_to_500 else RUN)
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
        plan['adapter_profile'] = BACKBONE_ATTENTION
        dataset_sha = datasets.pop()
        recipe = dict(vox_recipe_transfer.transfer_recipe('lora', dataset_sha), adapter_profile=BACKBONE_ATTENTION,
                      acoustic_recovery=vox_recovery.FOUR_PATCH, recovery_start_step=0)
        replay = dict(recipe=recipe, checkpoint_steps=[100, 250], audio_steps=[250], listening_plan=plan)
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

        local_resume = any((output / 'lora').glob('step-*/checkpoint.json'))
        print(f'[backbone] Target step {final_step}, weights at {replay["checkpoint_steps"]}, '
              f'{len(plan["cases"])} WAVs at {final_step}.', flush=True)
        vox_recipe_transfer.train(dataset, output, 'lora', final_step, resume=local_resume or parent is not None,
                                  fork_from=None if local_resume else parent, shared_features=source / 'features',
                                  recovery=vox_recovery.FOUR_PATCH, backbone_only=True,
                                  checkpoint_steps=tuple(replay['checkpoint_steps'][:-1]),
                                  expected_dataset_sha256=dataset_sha)
        checkpoint = output / 'lora' / f'step-{final_step:04d}'
        check_step(checkpoint, final_step, recipe)
        plan['checkpoints'] = listening.checkpoint_files(checkpoint)
        pack = output / 'audio' / checkpoint.name / 'backbone-complete'
        listening.render(pack, plan, audio_only=False)
        print(f'[backbone] Finished. Audio: {pack}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--continue-to-500', action='store_true',
                        help='Restore the completed backbone-only step 250 and train updates 251 through 500.')
    run(continue_to_500=parser.parse_args().continue_to_500)

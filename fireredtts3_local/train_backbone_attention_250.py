"""Train fresh backbone Q/K/V/O LoRA to 250 and render the matched listening set at 250."""
import json
from importlib.metadata import version
from pathlib import Path

from filelock import FileLock

from . import vox_attention_ablation as listening
from . import vox_recipe_levels, vox_recipe_transfer, vox_recovery
from .paths import ROOT
from .training_job import write_json
from .training_monitor import load_benchmark
from .training_objective import BACKBONE_ATTENTION
from .vox_recipe_compare import load_saved_base


SOURCE = ROOT / (vox_recipe_transfer.OUTPUT + '-levels')
PILOTS = ('comparison-0030-yrslvj2s', 'comparison-0030-ql21g14n')
RUN = 'lora-backbone-only-recovery-4patch-0250'


def run(source=SOURCE, dataset=ROOT / vox_recipe_levels.OUTPUT):
    source, dataset = Path(source).resolve(), Path(dataset).resolve()
    output = source / RUN
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
        manifest = output / 'replay.json'
        if manifest.exists():
            if json.loads(manifest.read_text()) != replay:
                raise ValueError(f'Existing run has different training or listening inputs: {manifest}')
        else:
            if (output / 'lora').exists() or (output / 'audio').exists():
                raise ValueError(f'Existing outputs have no matching replay metadata: {output}')
            write_json(manifest, replay)

        resume = any((output / 'lora').glob('step-*/checkpoint.json'))
        print(f'[backbone] 250 updates, weights at 100 and 250, {len(plan["cases"])} WAVs at 250.', flush=True)
        vox_recipe_transfer.train(dataset, output, 'lora', 250, resume=resume, shared_features=source / 'features',
                                  recovery=vox_recovery.FOUR_PATCH, backbone_only=True, checkpoint_steps=(100,),
                                  expected_dataset_sha256=dataset_sha)
        checkpoint = output / 'lora' / 'step-0250'
        record = json.loads((checkpoint / 'checkpoint.json').read_text())
        if record['step'] != 250 or record['recipe'] != recipe:
            raise ValueError(f'Unexpected final training checkpoint: {checkpoint}')
        plan['checkpoints'] = listening.checkpoint_files(checkpoint)
        pack = output / 'audio' / checkpoint.name / 'backbone-complete'
        listening.render(pack, plan, audio_only=False)
        print(f'[backbone] Finished. Audio: {pack}', flush=True)


if __name__ == '__main__':
    run()

"""Generate whole utterances on the gated 500-attempt LoRA with its original voice reference."""
import argparse
import json
from pathlib import Path

import soundfile as sf

from .history_probe import levels
from .model_manager import ModelManager
from .paths import MODEL_DIR, ROOT
from .training_data import file_digest, model_identity
from .training_job import write_json
from .training_monitor import SETTINGS, generate_case
from .vox_rise_gate import DEFAULT_CASE
from .workspace_paths import local_path

RUN = (ROOT / 'training/outputs/firered-vox-recipe-12-cap30m-levels'
       / 'lora-base-recovery-4patch-0500-rise20pct')
PHRASES = ROOT / 'examples/diagnostics/listening-ru-500.json'
OUTPUT = ROOT / 'training/eval/listening-rise20-step0500'


def listen(output, seeds):
    if len(set(seeds)) != len(seeds) or not seeds or any(not 0 <= n < 2**32 for n in seeds):
        raise ValueError('Supply distinct seeds between 0 and 4294967295')
    if output.exists():
        raise FileExistsError(f'{output} already exists. Choose a fresh folder with --output.')

    checkpoint = RUN / 'lora/step-0500'
    record = json.loads((checkpoint / 'checkpoint.json').read_text(encoding='utf-8'))
    recipe = record['recipe']
    gate = recipe['rise_gate']
    if (record['step'] != 500 or recipe['mode'] != 'lora' or gate['percent'] != 20
            or gate['settings'] != SETTINGS or SETTINGS['do_split']):
        raise ValueError(f'Expected the gated step-0500 LoRA and its unsplit inference settings: {checkpoint}')
    # The gate saved the actual reference bytes' hash and transcript alongside its training recipe.
    control = next(row for row in gate['cases'] if row['id'] == DEFAULT_CASE)
    ref = local_path(control['reference_audio']).resolve(strict=True)
    if file_digest(ref) != control['reference_sha256'] or not control['reference_text'].strip():
        raise ValueError(f'The original listening reference has changed: {ref}')
    if model_identity(MODEL_DIR) != recipe['base_sha256']:
        raise ValueError(f'Base weights differ from the checkpoint: {MODEL_DIR}')

    phrases = json.loads(PHRASES.read_text(encoding='utf-8'))
    report = dict(complete=False, checkpoint=str(checkpoint), attempt=record['step'],
                  accepted_updates=record['validation']['accepted_updates'], base_sha256=recipe['base_sha256'],
                  adapter_sha256=file_digest(checkpoint / 'adapter_model.safetensors'),
                  reference_audio=str(ref), reference_text=control['reference_text'],
                  reference_sha256=control['reference_sha256'], speaker_id=control.get('speaker_id'),
                  settings=SETTINGS, seeds=seeds, long_target_seconds=[40, 50], results=[])
    output.mkdir(parents=True)
    write_json(output / 'report.json', report)
    texts = [f'{group}/{name}\n{text}\n' for group, items in phrases.items() for name, text in items.items()]
    (output / 'texts.txt').write_text('\n'.join(texts), encoding='utf-8')

    manager = ModelManager()
    try:
        tts = manager.load_base(attention_backend='sdpa', adapter_path=str(checkpoint))
        for group, items in phrases.items():
            (output / group).mkdir()
            for name, text in items.items():
                for seed in seeds:
                    row = dict(text=text, language='Russian', seed=seed,
                               reference_audio=str(ref), reference_text=control['reference_text'])
                    samples, rate, generation = generate_case(tts, row, settings=gate['settings'])
                    measured = levels(samples, rate)
                    seconds = measured['seconds']
                    wav = Path(group) / f'{name}-seed{seed}.wav'
                    sf.write(output / wav, samples, rate, subtype='FLOAT')
                    result = dict(wav=wav.as_posix(), text=text, seed=seed, seconds=seconds,
                                  in_target_range=(40 <= seconds <= 50) if group == 'long' else None,
                                  finish_reason=generation['finish_reason'],
                                  rms_window_seconds=measured['rms_window_seconds'],
                                  end_minus_start_rms_db=measured['end_minus_start_rms_db'])
                    report['results'].append(result)
                    write_json(output / 'report.json', report)
                    print(f"[listen] {wav}  {seconds:.2f}s  finish={result['finish_reason']}  "
                          f"tail-head={result['end_minus_start_rms_db']:+.1f} dB", flush=True)
    finally:
        manager.unload()
    report['complete'] = True
    write_json(output / 'report.json', report)
    print(f"[listen] {len(report['results'])} WAVs in {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--seeds', nargs='+', type=int, default=[1234, 1235])
    args = parser.parse_args()
    listen(args.output.resolve(), args.seeds)


if __name__ == '__main__':
    main()

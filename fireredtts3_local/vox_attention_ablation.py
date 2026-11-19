"""Render FireRed listening comparisons with selected LoRA attention layers."""
import argparse
import csv
from importlib.metadata import version
import json
from pathlib import Path

from filelock import FileLock
from peft import PeftModel
from peft.tuners.lora.layer import LoraLayer
from peft.utils.other import ModulesToSaveWrapper
import soundfile as sf

from . import vox_listen, vox_recovery
from .history_probe import levels
from .model_manager import ModelManager
from .paths import MODEL_DIR, ROOT
from .sentence_diagnostics import ASR_SETTINGS, load_asr, transcript_errors
from .tail_contrast import contrast_windows
from .training_data import model_identity, file_digest
from .training_job import write_json
from .training_monitor import generate_case, load_benchmark, SETTINGS
from .training_objective import vox_attention_names
from .vox_recipe_transfer import BASE_SHA

OUTPUT = ROOT / 'training/eval/attention-resonant100-previous500'
DIT_OUTPUT = ROOT / 'training/eval/dit-cities-timofey-resonant100-previous500'
VARIANTS = ('backbone', 'dit', 'both')
STAGES = ('resonant-0100', 'previous-0500')


def route(core, variant):
    if variant not in VARIANTS or not isinstance(core, PeftModel) or set(core.peft_config) != {'default'}:
        raise ValueError('Expected one unmerged attention adapter')
    config = core.peft_config['default']
    if config.bias != 'none' or config.modules_to_save:
        raise ValueError('Extra trained parameters would invalidate this comparison')
    base = core.get_base_model()
    layers = {name: layer for name, layer in base.named_modules() if isinstance(layer, LoraLayer)}
    if set(layers) != set(vox_attention_names(base)) or any(isinstance(m, ModulesToSaveWrapper) for m in base.modules()):
        raise ValueError('Adapter targets do not match the Vox recipe backbone and DiT attention projections')
    if any(m.merged or m.active_adapters != ['default'] or set(m.lora_A) != {'default'}
           or set(m.lora_B) != {'default'} for m in layers.values()):
        raise ValueError('Expected exactly one unmerged default LoRA on every target')
    counts = {'backbone': 0, 'dit': 0}
    for name, layer in layers.items():
        group = 'backbone' if name.startswith('backbone_llm.') else 'dit'
        enabled = variant in ('both', group)
        layer.enable_adapters(enabled)
        if layer.disable_adapters == enabled:
            raise RuntimeError(f'Failed to route {name}')
        counts[group] += int(enabled)
    core.requires_grad_(False)
    core.eval()
    return counts


def make_plan(current_run, previous_run, *, dit_cities_timofey=False):
    data = {}
    reference = None
    for label, run, step, profile in zip(STAGES, (current_run, previous_run), (100, 500),
                                          (vox_recovery.RESONANT, vox_recovery.FOUR_PATCH), strict=True):
        path, record, control, ref = vox_listen.checkpoint_inputs(run, step)
        recipe = record['recipe']
        if recipe['acoustic_recovery'] != profile or recipe['base_sha256'] != BASE_SHA:
            raise ValueError(f'Unexpected training recipe or Base for {path}')
        current_ref = dict(reference_audio=str(ref), reference_text=control['reference_text'],
                           reference_sha256=control['reference_sha256'], language='Russian')
        if reference is not None and current_ref != reference:
            raise ValueError('The two runs used different reference audio or reference text')
        reference = current_ref
        data[label] = dict(path=str(path.resolve()), attempt=step, accepted=record['validation']['accepted_updates'],
                           dataset=recipe['dataset_sha256'], hashes={f:file_digest(path/f) for f in
                           ('adapter_config.json','adapter_model.safetensors','checkpoint.json')})
    if len({d['dataset'] for d in data.values()}) != 1 or model_identity(MODEL_DIR) != BASE_SHA:
        raise ValueError('Mismatched training selections or local Base model')
    if dit_cities_timofey:
        texts = {name: (ROOT / 'examples/diagnostics' / f'long-{name}.txt').read_text(encoding='utf-8').strip()
                 for name in ('cities', 'timofey')}
    else:
        texts = json.loads(vox_listen.PHRASES.read_text(encoding='utf-8'))['long']
    cases = [dict(id=f'{name}-seed{seed}', text=text, seed=seed)
             for name, text in texts.items() for seed in (1234, 1235)]
    plan = dict(schema='attention-ablation-v1', checkpoints=data, base_sha256=BASE_SHA,
                reference=reference, cases=cases, settings=SETTINGS, attention='sdpa', variants=list(VARIANTS),
                runtime={p:version(p) for p in ('torch','torchaudio','peft','transformers')})
    if dit_cities_timofey:
        plan.update(variants=['dit'], include_base=False)
    return plan


def series_plan(series, *, base_fingerprint=None):
    """Read the same rows and load the sibling training checkpoint used by vox_recipe_sweep."""
    step_dir = series.parent
    if not series.name.startswith('series-') or step_dir.parent.name != 'audio':
        raise ValueError('Expected RUN/audio/step-NNNN/series-N or its lora subdirectory')
    source = json.loads((series/'report.json').read_text(encoding='utf-8'))
    step = source['step']
    if not source['complete'] or step_dir.name != f'step-{step:04d}' or source['settings'] != SETTINGS:
        raise ValueError(f'Incomplete or incompatible listening series: {series}')
    checkpoint = step_dir.parent.parent/'lora'/step_dir.name
    record = json.loads((checkpoint/'checkpoint.json').read_text(encoding='utf-8'))
    recipe = record['recipe']
    if base_fingerprint is None:
        base_fingerprint = model_identity(MODEL_DIR)
    if (record['step'] != step or recipe != source['recipes']['lora'] or recipe['mode'] != 'lora'
            or recipe['base_sha256'] != BASE_SHA or base_fingerprint != BASE_SHA):
        raise ValueError(f'Checkpoint or Base differs from the saved series: {checkpoint}')
    cases, digest = load_benchmark(series/'benchmark.jsonl')
    originals = source['results']['lora']
    if digest != source['benchmark_sha256'] or len(cases) != len(originals):
        raise ValueError('Saved benchmark or reference differs from the series report')
    for index, (case, original) in enumerate(zip(cases, originals, strict=True)):
        filename = f'case-{index:03d}.wav'
        if (original['audio'] != f'lora/{filename}' or
                any(original[key] != case[key] for key in ('id', 'text', 'seed'))):
            raise ValueError(f'Case differs from the saved LoRA result: {filename}')
        case['filename'] = filename
    files = ('adapter_config.json', 'adapter_model.safetensors', 'checkpoint.json')
    weights = dict(path=str(checkpoint), hashes={name: file_digest(checkpoint/name) for name in files})
    return dict(schema='dit-series-v1', series=str(series), checkpoints={step_dir.name: weights},
                base_sha256=BASE_SHA, benchmark_sha256=digest, cases=cases, settings=source['settings'],
                variants=['dit'], include_base=False, attention='sdpa',
                runtime={p:version(p) for p in ('torch','torchaudio','peft','transformers')})


def phrase_cases(path, seeds):
    phrases = json.loads(path.read_text(encoding='utf-8'))
    cases = []
    for length in ('short', 'long'):
        for name, text in phrases[length].items():
            for seed in seeds:
                stem = f'{name}-seed{seed}'
                cases.append(dict(id=f'{length}-{stem}', text=text, seed=seed, filename=f'{length}/{stem}.wav'))
    return cases


def listening_plan(folder):
    """Use the checkpoint, active variant and first case's voice from a saved listening series."""
    if folder.name not in VARIANTS:
        raise ValueError('Expected a backbone, dit or both folder inside a saved listening series')
    plan = series_plan(folder.parent)
    saved = json.loads((folder/'report.json').read_text(encoding='utf-8'))['plan']
    if (saved['variants'] != [folder.name] or saved['checkpoints'] != plan['checkpoints']
            or saved['benchmark_sha256'] != plan['benchmark_sha256']):
        raise ValueError(f'Saved variant, checkpoint or benchmark differs from {folder}')
    first = plan['cases'][0]
    reference = {key: first[key] for key in ('reference_audio', 'reference_sha256', 'reference_text', 'language')}
    cases = phrase_cases(ROOT/'examples/diagnostics/listening-ru-calls.json', (1236, 1237))
    return dict(plan, schema='series-listening-calls-v1', variants=[folder.name], reference=reference,
                reference_case=first['id'], cases=cases)


def full_backbone_plan(step_dir):
    base_fingerprint = model_identity(MODEL_DIR)
    first = series_plan(step_dir/'series-1', base_fingerprint=base_fingerprint)
    second = series_plan(step_dir/'series-2', base_fingerprint=base_fingerprint)
    if first['checkpoints'] != second['checkpoints']:
        raise ValueError('Listening series use different checkpoints')
    controls = first['cases'] + second['cases']
    standard = {(f'{name}_seed{seed}', seed) for name in ('cities', 'timofey', 'question') for seed in (1234, 1235)}
    extra_cities = {(f'cities_seed{seed}', seed) for seed in range(1236, 1240)}
    for plan, expected in ((first, standard), (second, extra_cities)):
        if {(c['id'], c['seed']) for c in plan['cases']} != expected:
            raise ValueError(f"Unexpected listening cases in {plan['series']}; expected {sorted(expected)}")
    cases = [dict(c, filename=f"controls/{c['id']}.wav") for c in controls]
    # Match the original helpers: vox_listen used the gate control, listening_plan used series-2's first voice.
    gate_control = next(c for c in controls if c['id'] == vox_listen.DEFAULT_CASE)
    calls = ROOT/'examples/diagnostics/listening-ru-calls.json'
    for path, seeds, control in ((vox_listen.PHRASES, (1234, 1235), gate_control),
                                 (calls, (1236, 1237), second['cases'][0])):
        reference = {key: control[key] for key in ('reference_audio', 'reference_sha256', 'reference_text', 'language')}
        for item in phrase_cases(path, seeds):
            cases.append(dict(item, **reference, id=f"{path.stem}/{item['id']}",
                              filename=f"{path.stem}/{item['filename']}"))
    series = [{'path': plan['series'], 'benchmark_sha256': plan['benchmark_sha256']} for plan in (first, second)]
    return dict(schema='backbone-full-listening-v1', series=series, checkpoints=first['checkpoints'],
                base_sha256=first['base_sha256'], cases=cases, settings=first['settings'],
                variants=['backbone'], include_base=False, attention='sdpa', runtime=first['runtime'])


def generate(root, plan, report):
    results = report['results']
    stages = list(plan['checkpoints'])
    if plan.get('include_base', True):
        stages.insert(0, 'base')
    for label in stages:
        checkpoint = None if label == 'base' else plan['checkpoints'][label]['path']
        manager = ModelManager()
        tts = None
        try:
            for variant in (('base',) if checkpoint is None else plan['variants']):
                group = 'base' if checkpoint is None else f'{label}/{variant}'
                folder = Path() if 'series' in plan else Path(group)
                pending = []
                for case in plan['cases']:
                    name = (folder / case.get('filename', f"{case['id']}.wav")).as_posix()
                    saved = results.get(name)
                    if saved and (root/name).is_file() and file_digest(root/name) == saved['wav_sha256']:
                        continue
                    pending.append((case, name))
                if not pending:
                    continue
                if tts is None:
                    tts = manager.load_base(attention_backend='sdpa', adapter_path=checkpoint)
                    tts.tts_core.eval()
                counts = route(tts.tts_core, variant) if checkpoint else {'backbone':0, 'dit':0}
                for case, name in pending:
                    row = {**plan.get('reference', {}), **case}
                    samples, rate, run = generate_case(tts, row, settings=plan['settings'])
                    measured = levels(samples, rate)
                    scores = contrast_windows(samples, rate)
                    wav = root/name
                    wav.parent.mkdir(parents=True, exist_ok=True)
                    tmp = wav.with_suffix('.tmp.wav')
                    sf.write(tmp, samples, rate, subtype='FLOAT')
                    tmp.replace(wav)
                    results[name] = dict(group=group, case=case['id'], seconds=measured['seconds'],
                                         finish=run['finish_reason'], tail_minus_head_rms_db=measured['end_minus_start_rms_db'],
                                         **scores, active_layers=counts, wav_sha256=file_digest(wav))
                    write_json(root/'report.json', report)
                    print(f"[ablation] {name}  {measured['seconds']:.2f}s  finish={run['finish_reason']}  "
                          f"tail_contrast_db={scores['tail_contrast_db']}", flush=True)
        finally:
            tts = None
            manager.unload()


def recognize(root, plan, report):
    cases = {c['id']:c['text'] for c in plan['cases']}
    asr = dict(asr_model='large-v3', asr_settings=ASR_SETTINGS,
               asr_packages={p:version(p) for p in ('faster-whisper','ctranslate2','av')})
    recognizer, _ = load_asr(asr, root, print)
    for name, row in report['results'].items():
        if 'asr' in row:
            continue
        segments, _ = recognizer.transcribe(str(root/name), **ASR_SETTINGS)
        row['transcript'] = ' '.join(s.text.strip() for s in segments)
        row['asr'] = transcript_errors(cases[row['case']], row['transcript'])
        write_json(root/'report.json', report)
        print(f"[asr] {name}  WER={row['asr']['wer']:.3f}", flush=True)


def summary(root, results):
    fields = ('group','case','seconds','finish','tail_minus_head_rms_db','head_contrast_db','tail_contrast_db',
              'contrast_15_19_db','tail_contrast_delta_vs_base_db','wer','wer_delta_vs_base','missing_spans','wav')
    tmp = root/'summary.tmp'
    with tmp.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for name, row in sorted(results.items(), key=lambda item:(item[1]['case'], item[1]['group'])):
            base = results.get(f"base/{row['case']}.wav", {})
            tail, base_tail = row['tail_contrast_db'], base.get('tail_contrast_db')
            asr = row.get('asr', {})
            writer.writerow({**{k:row.get(k) for k in fields}, 'wav':name, 'wer':asr.get('wer'),
                             'wer_delta_vs_base':asr['wer']-base['asr']['wer'] if asr and 'asr' in base else None,
                             'tail_contrast_delta_vs_base_db':tail-base_tail if None not in (tail, base_tail) else None,
                             'missing_spans':' | '.join(s['text'] for s in asr.get('missing_spans', []))})
    tmp.replace(root/'summary.csv')


def run(root, current_run, previous_run, audio_only=False, *, dit_cities_timofey=False):
    plan = make_plan(current_run, previous_run, dit_cities_timofey=dit_cities_timofey)
    render(root, plan, audio_only)


def render(root, plan, audio_only):
    root.mkdir(parents=True, exist_ok=True)
    with FileLock(str(root/'.ablation.lock'), timeout=0):
        path = root/'report.json'
        if path.exists():
            report = json.loads(path.read_text(encoding='utf-8'))
            if report.get('plan') != plan:
                raise ValueError(f'Inputs changed; use a new output directory: {root}')
        else:
            if any(p.name != '.ablation.lock' for p in root.iterdir()):
                raise ValueError(f'Output already contains unrelated files: {root}')
            report = dict(plan=plan, results={})
        report.update(audio_complete=False, asr_complete=False)
        write_json(path, report)
        generate(root, plan, report)
        report['audio_complete'] = True
        write_json(path, report)
        summary(root, report['results'])
        if not audio_only:
            recognize(root, plan, report)
            summary(root, report['results'])
        report['asr_complete'] = all('asr' in row for row in report['results'].values())
        write_json(path, report)
        print(f"[ablation] {len(report['results'])} WAVs. Summary: {root/'summary.csv'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='Output directory (parent of the variant folders for multiple variants)')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--dit-cities-timofey', action='store_true',
                        help='Compare only the two DiT adapters on the original cities and timofey texts')
    mode.add_argument('--series', '--dit-series', type=Path,
                      help='Replay an existing series-N or series-N/lora folder (default variant: dit)')
    mode.add_argument('--listen-from', type=Path, metavar='VARIANT_DIR',
                      help='Generate new short and long calls using the voice and adapter of a saved series variant')
    mode.add_argument('--full-backbone', type=Path, metavar='AUDIO_STEP_DIR',
                      help='Render both original benchmark series and both complete listening sets with only backbone LoRA.')
    parser.add_argument('--variants', nargs='+', choices=VARIANTS,
                        help='LoRA components to enable in separate renders of --series')
    parser.add_argument('--audio-only', action='store_true', help='Save WAVs and spectral results; skip Whisper')
    args = parser.parse_args()
    if args.variants and not args.series:
        parser.error('--variants requires --series')
    if args.full_backbone:
        step_dir = args.full_backbone.resolve()
        render((args.output or step_dir/'backbone-complete').resolve(), full_backbone_plan(step_dir), args.audio_only)
        return
    if args.listen_from:
        folder = args.listen_from.resolve()
        render((args.output or folder/'listening').resolve(), listening_plan(folder), args.audio_only)
        return
    if args.series:
        series = args.series.resolve()
        if series.name == 'lora':
            series = series.parent
        plan = series_plan(series)
        variants = list(dict.fromkeys(args.variants or ['dit']))
        for variant in variants:
            output = series/variant
            if args.output:
                output = args.output/variant if len(variants) > 1 else args.output
            variant_plan = dict(plan, variants=[variant])
            if variant != 'dit':
                variant_plan['schema'] = 'attention-series-v1'
            print(f'[ablation] {variant} -> {output.resolve()}', flush=True)
            render(output.resolve(), variant_plan, args.audio_only)
        return
    previous = vox_listen.RUN
    output = args.output or (DIT_OUTPUT if args.dit_cities_timofey else OUTPUT)
    run(output.resolve(), previous.with_name(previous.name+'-resonant-w015'), previous, args.audio_only,
        dit_cities_timofey=args.dit_cities_timofey)


if __name__ == '__main__':
    main()

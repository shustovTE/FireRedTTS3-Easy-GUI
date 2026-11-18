"""Generate matched whole utterances and screen both transfer checkpoints against Base."""
import json
from pathlib import Path
import tempfile

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio

from .dialogs_dataset import lexical_key
from .history_probe import levels as waveform_levels
from .model_manager import ModelManager
from .paths import MODEL_DIR, ROOT
from .sentence_diagnostics import ASR_SETTINGS, transcript_errors
from .training_data import model_identity
from .training_monitor import SETTINGS, failures, generate_case, load_benchmark
from .vox_recipe_transfer import BASE_SHA, LR, configure, paired_data, read_rows
from .workspace_paths import local_path


def content_flags(result, base=None):
    reasons = failures(result, base)
    if any(span['words'] >= 3 for span in result['asr']['missing_spans']):
        reasons.append('asr_missing_three_consecutive_words')
    if base:
        if result['asr']['wer'] > base['asr']['wer'] + .05:
            reasons.append('wer_above_base_plus_0.05')
        if result['voice_similarity'] < base['voice_similarity'] - .02:
            reasons.append('voice_similarity_drop')
        if result['final_voice_similarity'] < base['final_voice_similarity'] - .02:
            reasons.append('final_voice_similarity_drop')
    return reasons


def default_benchmark(validation):
    voices = (ref for ref, target, _ in validation if target['dataset_id'] == 'dialogs')
    reference = min(voices, key=lambda ref: abs(ref['duration'] - 6.))
    texts = {'question': 'Ничего страшного. Подскажите хотя бы город, чтобы я понимала, работаем ли мы в вашем регионе?'}
    for name in ('cities', 'timofey'):
        texts[name] = (ROOT / 'examples/diagnostics' / f'long-{name}.txt').read_text().strip()
    return [dict(id=f'{name}_seed{seed}', text=text, reference_audio=reference.get('unlevelled_audio', reference['audio']),
                 reference_text=reference['text'], speaker_id=reference['speaker_id'], language='Russian', seed=seed)
            for name, text in texts.items() for seed in (1234, 1235)]


def compare(dataset, output, step, benchmark, asr, speaker_model, against=None):
    from faster_whisper import WhisperModel
    from transformers import AutoFeatureExtractor, WavLMForXVector
    splits, _, selection = paired_data(dataset)
    if model_identity(MODEL_DIR) != BASE_SHA:
        raise ValueError('Compare with the original training base')
    checkpoints = {mode: output / mode / f'step-{step:04d}' for mode in LR}
    recipes = {}
    for mode, path in checkpoints.items():
        record = json.loads((path / 'checkpoint.json').read_text())
        recipe = record['recipe']
        if (record['step'], recipe['base_sha256'], recipe['dataset_sha256'], recipe['mode']) != (
                step, BASE_SHA, selection['selection_sha256'], mode):
            raise ValueError(f'Checkpoint/data mismatch for {mode}')
        recipes[mode] = recipe
    previous = None
    if against is not None:
        if 'original_selection_sha256' not in selection:
            raise ValueError('--against requires the levelled dataset')
        previous = json.loads(against.read_text(encoding='utf-8'))
        if not previous['complete'] or previous['step'] != step:
            raise ValueError('Expected a completed comparison at the same step')
        for mode, recipe in recipes.items():
            expected = dict(recipe, dataset_sha256=selection['original_selection_sha256'])
            if previous['recipes'][mode] != expected:
                raise ValueError(f'Original comparison recipe/selection mismatch for {mode}')
        benchmark = against.parent / 'benchmark.jsonl'
    root = Path(tempfile.mkdtemp(prefix=f'comparison-{step:04d}-', dir=output))
    if benchmark is None:
        rows = default_benchmark(splits['validation'])
    else:
        benchmark = local_path(benchmark).resolve()
        rows = read_rows(benchmark)
        for row in rows:
            row['reference_audio'] = str((benchmark.parent / local_path(row['reference_audio'])).resolve())
    case_file = root / 'benchmark.jsonl'
    case_file.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    cases, fingerprint = load_benchmark(case_file)
    train_sha = {r['audio_sha256'] for pair in splits['train'] for r in pair[:2]}
    train_sha.update(r['unlevelled_audio_sha256'] for pair in splits['train'] for r in pair[:2]
                     if 'unlevelled_audio_sha256' in r)
    train_words = {lexical_key(pair[1]['text']) for pair in splits['train']}
    if any(row['reference_sha256'] in train_sha or lexical_key(row['text']) in train_words for row in cases):
        raise ValueError('Evaluation reference or target text occurs in training')
    if previous and (previous['benchmark_sha256'], previous['settings'], previous['asr'], previous['speaker_model']) != (
            fingerprint, SETTINGS, asr, speaker_model):
        raise ValueError('Reference audio, cases, inference settings or scorers differ from the original comparison')
    (root / 'cases.json').write_text(json.dumps(cases, ensure_ascii=False, indent=2))
    recognizer = WhisperModel(asr, device='cpu', compute_type='int8', cpu_threads=4)
    extractor = AutoFeatureExtractor.from_pretrained(speaker_model)
    scorer = WavLMForXVector.from_pretrained(speaker_model).eval()
    def embed(wav, sr):
        data = torchaudio.functional.resample(wav.float(), sr, 16000).mean(0)
        inputs = extractor(data.numpy(), sampling_rate=16000, return_tensors='pt')
        with torch.inference_mode():
            return F.normalize(scorer(**inputs).embeddings, dim=-1)
    refs = {}
    baseline = {}
    report = dict(step=step, complete=False, benchmark_sha256=fingerprint, settings=SETTINGS,
                  recipes=recipes, asr=asr, speaker_model=speaker_model, results={})
    if previous:
        report['compared_to'] = str(against.resolve())
    for mode in ('base', *LR):
        folder = root / mode
        folder.mkdir()
        manager = ModelManager()
        results = []
        try:
            tts = manager.load_base(attention_backend='sdpa', torch_compile=False)
            if mode != 'base':
                tts.tts_core = configure(tts.tts_core, mode, checkpoints[mode])
            tts.tts_core.eval()
            for i, row in enumerate(cases):
                samples, rate, result = generate_case(tts, row)
                if not len(samples) or not np.isfinite(samples).all():
                    raise FloatingPointError(f'Invalid generated WAV for {mode}/{row["id"]}')
                result.update(waveform_levels(samples, rate))
                path = folder / f'case-{i:03d}.wav'
                sf.write(path, samples, rate, subtype='FLOAT')
                segments, _ = recognizer.transcribe(str(path), **ASR_SETTINGS)
                transcript = ' '.join(segment.text for segment in segments)
                key = row['reference_sha256']
                if key not in refs:
                    refs[key] = embed(*torchaudio.load(row['reference_audio']))
                wav = torch.from_numpy(samples)[None]
                # Last third, with a minimum of one second where available.
                n = min(wav.shape[-1], max(rate, wav.shape[-1] // 3))
                result.update(id=row['id'], seed=row['seed'], audio=str(path.relative_to(root)), text=row['text'],
                              transcript=transcript, asr=transcript_errors(row['text'], transcript),
                              voice_similarity=float((embed(wav, rate)*refs[key]).sum()),
                              final_voice_similarity=float((embed(wav[:, -n:], rate)*refs[key]).sum()))
                if not np.isfinite([result['voice_similarity'], result['final_voice_similarity']]).all():
                    raise FloatingPointError('Nonfinite speaker score')
                result['flags'] = content_flags(result, baseline.get(row['id']))
                if previous:
                    before = next(r for r in previous['results'][mode] if r['id'] == row['id'])
                    result['delta_vs_unlevelled'] = {
                        'wer': result['asr']['wer'] - before['asr']['wer'],
                        **{k: result[k] - before[k] for k in ('end_minus_start_rms_db', 'voice_similarity',
                                                               'final_voice_similarity')}}
                if mode == 'base':
                    baseline[row['id']] = result
                results.append(result)
                print(f'[compare] {mode} {row["id"]}: {result["flags"]}', flush=True)
        finally:
            manager.unload()
        report['results'][mode] = results
        (root / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    report.update(complete=True, listening_required=True, passed_automatic_checks={
        mode: all(not r['flags'] for r in rows) for mode, rows in report['results'].items()})
    (root / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return root / 'report.json'

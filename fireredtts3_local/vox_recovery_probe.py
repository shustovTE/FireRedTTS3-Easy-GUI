"""Paired whole-utterance continuations from a shared clean or metallic target prefix."""
from difflib import SequenceMatcher
import json
from pathlib import Path
import tempfile

import numpy as np
import soundfile as sf
import torch
import torchaudio

from .dialogs_dataset import lexical_key
from .history_probe import cpu_audio, levels, read_audio
from .model_manager import ModelManager
from .sentence_diagnostics import ASR_SETTINGS, transcript_errors, words
from .training_data import file_digest
from .training_job import write_json
from .training_monitor import SETTINGS
from .training_run import isolated_rng
from .vox_recovery import corrupt_tail


TEXTS = {
    'visit': ('Давайте уточним время выезда. В четверг инженер сможет приехать к вам после обеда, '
              'осмотреть ванную и кухню, проверить проводку и замерить обе комнаты. Затем он пришлёт '
              'вам подробную смету. Какой промежуток удобнее, с двух до четырёх или ближе к вечеру?'),
    'delivery': ('Я посмотрела расписание доставки. Машина уже выехала со склада, но водитель сначала '
                 'заедет на соседнюю улицу. Пожалуйста, оставьте ему номер подъезда и предупредите охрану. '
                 'Если лифт ещё выключен, коробки можно временно оставить у консьержа. Вам так подойдёт?'),
    'handover': ('Хочу проверить, всё ли мы записали верно. В спальне остаётся старый паркет, в гостиной '
                 'меняем розетки, а для коридора подбираем светильник потеплее. Образцы привезём на '
                 'следующую встречу. Если оттенок понравится только при дневном свете, решение можно '
                 'принять утром. Кому отправить фотографии и список материалов?'),
}
SEEDS = (1234, 1235)
DAMAGE = dict(gain_db=6., delay_ms=3., comb_mix=.50, ramp_patches=8)
MAX_PATCHES = 400


def select_voice(splits):
    train_hashes = {r['audio_sha256'] for pair in splits['train'] for r in pair[:2]}
    train_text = {lexical_key(row['text']) for _, row, _ in splits['train']}
    if any(lexical_key(text) in train_text for text in TEXTS.values()):
        raise ValueError('A probe continuation occurs in training')
    eligible = [(r, p) for r, p, _ in splits['validation'] if p['dataset_id'] == 'dialogs'
                and r['speaker_id'] == p['speaker_id'] and r['audio_sha256'] != p['audio_sha256']
                and not train_hashes.intersection((r['audio_sha256'], p['audio_sha256']))]
    if not eligible:
        raise ValueError('The continuation test needs two distinct held-out recordings of one Dialogs voice')
    ref, prefix = min(eligible, key=lambda pair: (abs(pair[1]['duration'] - 5.),
                                                  abs(pair[0]['duration'] - 6.), pair[1]['utt_id']))
    fields = ('audio', 'audio_sha256', 'text', 'speaker_id')
    return {key: {k: row[k] for k in fields} for key, row in (('reference', ref), ('prefix', prefix))}


@torch.inference_mode()
def prepare_history(tts, voice, root):
    codec = tts.redae
    rate = codec.sample_rate
    stride = tts.tts_core.patch_size * codec.downsample_rate
    _, reference, _ = read_audio(tts, voice['reference'])
    source, clean, left = read_audio(tts, voice['prefix'])
    altered = corrupt_tail(clean.cpu(), stride * DAMAGE['ramp_patches'], DAMAGE['gain_db'],
                           max(1, round(rate * DAMAGE['delay_ms'] / 1000)), DAMAGE['comb_mix'])
    if torch.equal(altered, clean.cpu()) or not torch.isfinite(altered).all():
        raise ValueError('The intervention produced an identical or nonfinite waveform')
    variants = {}
    for name, wave in (('clean', clean), ('metal', altered)):
        samples = cpu_audio(wave[:, left:])
        path = root / f'prefix-{name}.wav'
        if path.exists():
            actual, sr = sf.read(path, dtype='float32')
            if sr != rate or not np.array_equal(samples, actual):
                raise ValueError(f'Shared prefix changed: {path}')
        else:
            tmp = path.with_suffix('.tmp.wav')
            sf.write(tmp, samples, rate, subtype='FLOAT')
            tmp.replace(path)
        variants[name] = codec.encode(wave.to(tts.device), rate).float()
    speaker = tts.spk_extractor(reference, rate).to(tts.device)
    return dict(reference=codec.encode(reference, rate).float(), speaker=speaker,
                prefixes=variants, prefix_samples=source.shape[1], left=left, rate=rate)


@torch.inference_mode()
def continuation(tts, history, tokens, variant, seed):
    """The prefill feeds both backbone and DiT. The native loop generates every remaining patch."""
    core = tts.tts_core
    prefix = history['prefixes'][variant]
    observed = torch.cat([history['reference'], prefix], dim=1)
    consumed = prefix.shape[1] // core.patch_size
    budget = MAX_PATCHES - consumed
    if not 0 < budget < MAX_PATCHES:
        raise ValueError('Recorded prefix outside native target horizon')
    with isolated_rng(seed):
        latents = core.generate(spk_emb=history['speaker'], text_tokens=tokens, prompt_latents=observed,
                                n_timesteps=SETTINGS['n_timesteps'], inference_cfg=SETTINGS['inference_cfg'],
                                stop_threshold=SETTINGS['stop_threshold'], min_gen_steps=max(0, 6-consumed),
                                max_gen_steps=budget)
    added = latents.shape[1] - observed.shape[1]
    if added < 0 or added % core.patch_size or added // core.patch_size > budget:
        raise ValueError('Invalid native continuation length')
    wave, rate = tts.redae.decode(latents)
    stride = tts.redae.downsample_rate
    if rate != history['rate'] or wave.shape[1] != latents.shape[1] * stride:
        raise ValueError('Decoded waveform does not match the supplied patch grid')
    start = history['reference'].shape[1] * stride + history['left']
    return cpu_audio(wave[:, start:]), 'max_patches' if added // core.patch_size == budget else 'stop'


def score_suffix(samples, rate, text, recognizer):
    timed = []
    transcript = ''
    if len(samples):
        audio = torchaudio.functional.resample(torch.from_numpy(samples.copy()), rate, 16000).numpy()
        segments, _ = recognizer.transcribe(audio, **ASR_SETTINGS)
        segments = list(segments)
        transcript = ' '.join(s.text.strip() for s in segments)
        for segment in segments:
            for w in segment.words or []:
                timed.append(dict(word=w.word.strip(), start=w.start, end=w.end))
    return dict(transcript=transcript, asr=transcript_errors(text, transcript), words=timed,
                levels=levels(samples, rate) if len(samples) else None)


def word_levels(samples, rate, text, timed):
    # Compare the same recognized words even when free continuations differ in pace.
    tokens = [(token, item) for item in timed for token in words(item['word'])]
    matcher = SequenceMatcher(None, words(text), [token for token, _ in tokens], autojunk=False)
    by_index = {}
    frame = round(.025 * rate)
    window = np.hanning(frame)
    freqs = np.fft.rfftfreq(frame, 1/rate)
    edges = np.geomspace(125., min(8000., rate/2), 9)
    for start, offset, count in matcher.get_matching_blocks():
        for j in range(count):
            token, item = tokens[offset+j]
            audio = samples[max(0, round(rate*item['start'])):round(rate*item['end'])]
            if len(audio) < round(.03 * rate):
                continue
            rms = float(np.sqrt(np.mean(audio.astype('float64')**2)))
            frames = np.lib.stride_tricks.sliding_window_view(audio, frame)[::max(1, frame//2)]
            power = np.mean(np.abs(np.fft.rfft(frames * window))**2, axis=0)
            bands = [power[(freqs >= low) & (freqs < high)].sum() for low, high in zip(edges[:-1], edges[1:])]
            band_db = 10 * np.log10(np.maximum(bands, 1e-12))
            by_index[start+j] = dict(word=token, start=item['start'], end=item['end'],
                                     rms_dbfs=20*np.log10(max(1e-12, rms)), shape_db=band_db-band_db.mean())
    return by_index


def run_pair(tts, history, voice, case, seed, root, directory, recognizer):
    text = TEXTS[case]
    effective, language, sentences = tts._apply_frontend(text=voice['prefix']['text'] + ' ' + text,
                                                          language='Russian', do_split=False)
    if len(sentences) != 1:
        raise ValueError('Expected one unsplit target')
    tokens = tts._tokenize_text(f'<|{language}|><|sot|>{voice["reference"]["text"]}{effective}<|eot|>')
    n, rate = history['prefix_samples'], history['rate']
    results, aligned = {}, {}
    with tempfile.TemporaryDirectory(dir=directory.parent) as tmp:
        stage = Path(tmp) / 'pair'
        stage.mkdir()
        for variant in ('clean', 'metal'):
            samples, finish = continuation(tts, history, tokens, variant, seed)
            path = stage / f'{variant}.wav'
            sf.write(path, samples, rate, subtype='FLOAT')
            suffix = samples[n:]
            result = score_suffix(suffix, rate, text, recognizer)
            aligned[variant] = word_levels(suffix, rate, text, result['words'])
            results[variant] = dict(result, finish=finish, seconds_after_prefix=len(suffix)/rate,
                                    audio=str((directory/path.name).relative_to(root)), sha256=file_digest(path))
        gaps = []
        for i in sorted(aligned['clean'].keys() & aligned['metal'].keys()):
            a, b = aligned['clean'][i], aligned['metal'][i]
            gaps.append(dict(word_number=i+1, word=a['word'], clean_start=a['start'], metal_start=b['start'],
                             metal_minus_clean_rms_db=b['rms_dbfs'] - a['rms_dbfs'],
                             spectral_shape_distance_db=float(np.sqrt(np.mean((b['shape_db']-a['shape_db'])**2)))))
        row = dict(case=case, seed=seed, continuation_text=text, prefix_seconds=n/rate, variants=results,
                   matched_words=gaps, wer_delta=results['metal']['asr']['wer'] - results['clean']['asr']['wer'])
        write_json(stage / 'pair.json', row)
        stage.rename(directory)
    return row


def probe_stage(root, label, adapter, voice, recipe):
    """Called under the sweep's run lock. Completed pairs survive a restart during scoring."""
    from faster_whisper import WhisperModel
    folder = root / label
    folder.mkdir(parents=True, exist_ok=True)
    plan = dict(schema='matched-prefix-continuation-v1', voice=voice, texts=TEXTS, seeds=list(SEEDS),
                settings=SETTINGS, damage=DAMAGE, max_target_patches=MAX_PATCHES, asr='large-v3',
                asr_settings=ASR_SETTINGS, base_sha256=recipe['base_sha256'],
                dataset_sha256=recipe['dataset_sha256'], stage=label,
                checkpoint={f: file_digest(adapter/f) for f in
                            ('checkpoint.json', 'adapter_config.json', 'adapter_model.safetensors')} if adapter else None)
    plan_file = folder / 'plan.json'
    if plan_file.exists() and json.loads(plan_file.read_text()) != plan:
        raise ValueError(f'Continuation test inputs changed: {plan_file}')
    write_json(plan_file, plan)
    manager, tts, recognizer, history = ModelManager(), None, None, None
    rows = []
    try:
        for case in TEXTS:
            for seed in SEEDS:
                directory = folder / f'{case}-seed{seed}'
                if directory.exists():
                    row = json.loads((directory/'pair.json').read_text())
                    for result in row['variants'].values():
                        if file_digest(root/result['audio']) != result['sha256']:
                            raise ValueError(f'Existing continuation audio changed: {directory}')
                else:
                    if tts is None:
                        tts = manager.load_base(attention_backend='sdpa', torch_compile=False,
                                                adapter_path=str(adapter) if adapter else None)
                        for module in (tts.tts_core, tts.redae, tts.spk_extractor):
                            module.eval()
                        history = prepare_history(tts, voice, root)
                        recognizer = WhisperModel(plan['asr'], device='cpu', compute_type='int8', cpu_threads=4)
                    print(f'[prefix-test] {label} {case} seed={seed}', flush=True)
                    row = run_pair(tts, history, voice, case, seed, root, directory, recognizer)
                rows.append(row)
    finally:
        tts = None
        manager.unload()
    report = folder / 'report.json'
    write_json(report, dict(complete=True, plan=plan, pairs=rows, times_relative_to='start of generated continuation',
                            note='RMS and ASR are descriptive. Judge metallic timbre on matched WAVs.'))
    print(f'[prefix-test] {report}', flush=True)
    return report

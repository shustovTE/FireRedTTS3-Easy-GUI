"""Whole text versus complete sentences, with independent ASR completeness checks."""
from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
from pathlib import Path
import re
import signal
import zipfile

from filelock import FileLock

from . import cfg_diagnostics as shared
from .training_data import file_digest, plain_text
from .training_job import write_json, read_status

SCHEMA = 'native-sentence-diagnostics-v1'
DEFAULT_OUTPUT = 'training/eval/joint-10-sentence-diagnostics'
DEFAULT_JOB = 'training/jobs/joint-10-sentences'
ASR_SETTINGS = dict(language='ru', beam_size=5, temperature=0., vad_filter=False,
                    condition_on_previous_text=False, word_timestamps=True)
VARIANTS = (('base', ()), ('backbone_only', ('backbone',)), ('dit_only', ('dit',)))


def split_sentences(text):
    """Conservative punctuation boundaries; preserve every normalized character.

    Decimal points, initials and common Russian abbreviations stay inside a
    segment. This is not a general linguistic sentence detector or length cap.
    """
    text = plain_text(text)
    endings = r'[.!?]+[»”"\')\]]*'
    if not text or not re.search(endings+r'$', text):
        raise ValueError('Diagnostic text must end with sentence punctuation')
    abbreviations = {'г', 'гг', 'ул', 'д', 'стр', 'им', 'рис', 'т', 'е', 'д', 'п', 'др', 'пр', 'см'}
    result, start = [], 0
    for match in re.finditer(endings+r'(?=\s|$)', text):
        prefix = text[:match.start()]
        word = re.search(r'(\w+)$', prefix)
        if match.group() == '.' and match.end() < len(text) and word:
            token = word.group()
            if token.lower() in abbreviations or (len(token) == 1 and token.isupper()):
                continue
        end = match.end()
        result.append(text[start:end].strip())
        start = end
    if text[start:].strip():
        result.append(text[start:].strip())
    if not result or ' '.join(result) != text:
        raise ValueError('Sentence splitting lost or changed text')
    return result


def make_plan(run=shared.DEFAULT_RUN, output=DEFAULT_OUTPUT, step=143,
              cases=shared.DEFAULT_CASES, asr_model='large-v3', pause_seconds=.2):
    if not isinstance(asr_model, str) or not asr_model.strip():
        raise ValueError('Require an ASR model name or local CTranslate2 model directory')
    if not 0 <= pause_seconds <= 1:
        raise ValueError('Sentence pause must be between zero and one second')
    plan = shared.make_plan(run, output, steps=(step,), cfgs=(2.,), cases=cases)
    control = json.loads(Path(plan['previous_reports']['0']['path']).read_text())
    ids = [row['id'] for row in control['results']]
    if len(ids) != len(set(ids)) or not set(cases) <= set(ids):
        raise ValueError('Previous base report has missing or duplicate cases')
    originals = plan['cases']
    base, trained = plan['stages']
    expanded = []
    for i, row in enumerate(originals):
        text = plain_text(row['text'])
        sentences = split_sentences(text)
        for mode, parts in [('whole', [text]), ('sentences', sentences)]:
            for index, part in enumerate(parts):
                expanded.append(dict(row, id=f'case-{i:02d}-{mode}-{index:03d}', text=part,
                    source_case_id=row['id'], mode=mode, sentence_index=index,
                    comparison_id=row['id'] if mode=='whole' else None))
    plan.update(schema=SCHEMA, source_step=step, original_cases=originals, cases=expanded,
                stages=[dict(base if name=='base' else trained, name=name, active_components=list(active),
                             comparison_step=0 if name=='base' else None) for name, active in VARIANTS],
                total_cases=len(VARIANTS)*len(expanded), total_comparisons=len(VARIANTS)*len(originals)*2,
                pause_seconds=float(pause_seconds), asr_model=asr_model, asr_settings=ASR_SETTINGS,
                asr_device='cpu', asr_compute_type='int8', asr_cpu_threads=4,
                asr_packages={name: importlib.metadata.version(name) for name in ('faster-whisper','ctranslate2','av')})
    return plan


def words(text):
    return re.findall(r'\w+', plain_text(text).lower().replace('ё', 'е'))


def transcript_errors(reference, hypothesis):
    """Levenshtein edits and contiguous deleted spans, not a semantic score."""
    a, b = words(reference), words(hypothesis)
    if not a:
        raise ValueError('Expected nonempty reference transcript')
    costs = [list(range(len(b)+1))] + [[i]+[0]*len(b) for i in range(1,len(a)+1)]
    for i, left in enumerate(a, 1):
        for j, right in enumerate(b, 1):
            costs[i][j] = min(costs[i-1][j]+1, costs[i][j-1]+1, costs[i-1][j-1]+(left!=right))
    i, j = len(a), len(b)
    operations = []
    while i or j:
        if i and j and costs[i][j] == costs[i-1][j-1]+(a[i-1]!=b[j-1]):
            operations.append(('match' if a[i-1]==b[j-1] else 'substitution', i-1)); i-=1; j-=1
        elif i and costs[i][j] == costs[i-1][j]+1:
            operations.append(('deletion', i-1)); i-=1
        else:
            operations.append(('insertion', None)); j-=1
    operations.reverse()
    counts = {name: sum(op==name for op,_ in operations) for name in ('match','substitution','deletion','insertion')}
    deleted = {index for op,index in operations if op=='deletion'}
    spans, start = [], None
    for i in range(len(a)+1):
        if i in deleted and start is None:
            start = i
        elif i not in deleted and start is not None:
            spans.append(dict(start_word=start, end_word=i, text=' '.join(a[start:i]), words=i-start)); start=None
    return dict(wer=costs[-1][-1]/len(a), reference_words=len(a), hypothesis_words=len(b),
                **counts, missing_spans=spans,
                note='ASR edit alignment can be ambiguous; deletions are suspected omissions, not human verification.')


def verify_artifacts(root, artifacts):
    for name, expected in artifacts.items():
        path = (root/name).resolve()
        if not path.is_relative_to(root) or file_digest(path) != expected:
            raise ValueError(f'Diagnostic audio changed: {name}')


def load_asr(plan, root, log):
    from faster_whisper import WhisperModel
    from faster_whisper.utils import download_model
    saved = root/'asr-runtime.json'
    prior = json.loads(saved.read_text()) if saved.exists() else None
    log('[sentence-asr] Loading ASR on CPU; checking model files (first use may download weights)')
    path = Path(prior['path']) if prior else Path(plan['asr_model']).expanduser()
    if not prior and not path.is_dir():
        path = Path(download_model(plan['asr_model']))
    path = path.resolve()
    files = [p for p in sorted(path.iterdir()) if p.is_file() and
             (p.name in ('model.bin','config.json','tokenizer.json','preprocessor_config.json') or p.name.startswith('vocabulary.'))]
    if not {'model.bin','config.json','tokenizer.json'} <= {p.name for p in files}:
        raise ValueError('ASR requires complete local CTranslate2 model, config and tokenizer files')
    public = dict(model=plan['asr_model'], files={p.name: file_digest(p) for p in files},
                  packages=plan['asr_packages'], settings=plan['asr_settings'], device='cpu', compute_type='int8')
    identity = dict(path=str(path), public=public)
    if prior and identity != prior:
        raise ValueError('ASR weights/runtime changed; do not mix recognition results')
    if not prior:
        write_json(saved, identity)
    recognizer = WhisperModel(str(path), device='cpu', compute_type='int8', cpu_threads=4,
                              num_workers=1, local_files_only=True)
    return recognizer, public


def assemble(root, plan, stage, original, mode, generated):
    import numpy as np
    import soundfile as sf
    from .native_comparison import signal_levels
    parts = [r for r in generated if r['stage']==stage and r['source_case_id']==original['id'] and r['mode']==mode]
    parts.sort(key=lambda r: r['sentence_index'])
    expected = [r for r in plan['cases'] if r['source_case_id']==original['id'] and r['mode']==mode]
    if [r['id'] for r in parts] != [r['id'] for r in expected]:
        raise ValueError('Missing or reordered sentence generations')
    arrays, boundaries, offset, rate = [], [], 0, None
    for i, part in enumerate(parts):
        values, sr = sf.read(root/part['float_audio'], dtype='float32')
        if values.ndim != 1 or (rate is not None and sr != rate):
            raise ValueError('Sentence sample rates/channels differ')
        rate = sr
        if i:
            pause = np.zeros(round(plan['pause_seconds']*rate), dtype=np.float32)
            arrays.append(pause); offset += len(pause)
        boundaries.append(dict(index=i, text=part['text'], start_seconds=offset/rate,
            end_seconds=(offset+len(values))/rate, source_key=part['key'], finish_reason=part['finish_reason']))
        arrays.append(values); offset += len(values)
    samples = np.concatenate(arrays)
    levels = signal_levels(samples, rate)
    n = min(10*rate, len(samples))
    rms = lambda x: 20*np.log10(max(np.sqrt(np.mean(x.astype(np.float64)**2)), 1e-12))
    levels['end_minus_start_rms_db'] = float(rms(samples[-n:])-rms(samples[:n]))
    folder = root/'comparison-audio'/stage/mode
    folder.mkdir(parents=True, exist_ok=True)
    index = plan['original_cases'].index(original)
    paths = [folder/f'case-{index:02d}.float.wav', folder/f'case-{index:02d}.wav']
    gain = min(1., .98/max(levels['peak'], 1e-12))
    for path, values, subtype in zip(paths, (samples,samples*gain), ('FLOAT','PCM_16')):
        temporary = path.with_suffix('.tmp')
        sf.write(temporary, values, rate, format='WAV', subtype=subtype); temporary.replace(path)
    return dict(key=f'{stage}/{mode}/{original["id"]}', stage=stage, mode=mode, id=original['id'],
        text=plain_text(original['text']), seed=original['seed'], reference_sha256=original['reference_sha256'],
        **levels, pcm_gain=gain,
        parts=boundaries, parts_count=len(parts),
        added_silence_seconds=(len(parts)-1)*round(plan['pause_seconds']*rate)/rate,
        finish_reason='stop' if all(p['finish_reason']=='stop' for p in parts) else 'segment_limit',
        audio=str(paths[1].relative_to(root)), float_audio=str(paths[0].relative_to(root)),
        artifacts={str(p.relative_to(root)):file_digest(p) for p in paths})


def persist_comparison(root, plan, generation, identity, results):
    index = {(r['stage'],r['mode'],r['id']):r for r in results}
    for row in results:
        base = index.get(('base',row['mode'],row['id']))
        whole = index.get((row['stage'],'whole',row['id']))
        row['wer_delta_vs_base_same_mode'] = row['asr']['wer']-base['asr']['wer'] if base else None
        row['wer_delta_vs_whole_same_weights'] = row['asr']['wer']-whole['asr']['wer'] if whole else None
    controls = [r for r in generation['results'] if r['stage']=='base' and r['mode']=='whole']
    report = dict(schema=SCHEMA, plan_sha256=shared.signature(plan), generation_sha256=shared.signature(generation),
        source_step=plan['source_step'], generation_settings=plan['settings'],
        components=[{k:s[k] for k in ('name','active_components','adapter_fingerprint') if k in s} for s in plan['stages']],
        complete=len(results)==plan['total_comparisons'], completed_cases=len(results), total_cases=plan['total_comparisons'],
        asr=identity, results=results,
        reproduction_controls_complete=len(controls)==len(plan['original_cases']) and all(r['previous_monitor'] for r in controls),
        reproduction_review_required=any(r['previous_monitor'] and r['previous_monitor']['reproduction_review_required'] for r in controls),
        note='ASR completeness screening, not perceptual approval. Whole vs sentences resets context and seed per sentence; '
             'same original reference every time. Joined audio has fixed pauses, no crossfade or per-sentence normalization.')
    write_json(root/'comparison.json', report)
    fields = ['stage','mode','id','seconds','parts_count','added_silence_seconds','finish_reason','end_minus_start_rms_db',
              'wer','deletion','substitution','insertion','wer_delta_vs_base_same_mode','wer_delta_vs_whole_same_weights','audio']
    temporary = root/'comparison.csv.tmp'
    with temporary.open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore');writer.writeheader()
        writer.writerows({**r,**r['asr']} for r in results)
    temporary.replace(root/'comparison.csv')
    lines = ['# Whole text versus sentences', '', f"ASR completed {len(results)}/{plan['total_comparisons']} outputs.", '',
             '| Weights | Mode | Case | Seconds | Parts | Finish | WER | Deleted words | RMS drift dB |',
             '|---|---|---|---:|---:|---|---:|---:|---:|']
    for r in results:
        lines.append(f"| {r['stage']} | {r['mode']} | {r['id']} | {r['seconds']:.2f} | {r['parts_count']} | "
                     f"{r['finish_reason']} | {100*r['asr']['wer']:.1f}% | {r['asr']['deletion']} | {r['end_minus_start_rms_db']:+.2f} |")
    lines += ['', f"ASR: {plan['asr_model']}, CPU int8; same recognizer/settings for every output; no transcript prompt.",
              f"Sentences use the original voice reference and reset to the case seed; fixed {plan['pause_seconds']:g}s silence between parts.",
              'Seconds include added silence. RMS depends on content/pauses and is not a perceptual metallic-timbre score.',
              'Inspect asr.text, asr.segments and asr.missing_spans in comparison.json. ASR may make mistakes.',
              'No automatic winner or deployment approval. Compare completeness, timbre, prosody and voice identity together.']
    (root/'comparison.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return report


def score(plan, log=print):
    shared.verify_plan(plan)
    root = Path(plan['output'])
    stopping = False
    def stop(_signal, _frame):
        nonlocal stopping
        stopping=True
        log('[sentence-asr] Stop requested; finish this WAV and save')
    handlers={s:signal.signal(s,stop) for s in (signal.SIGINT,signal.SIGTERM)}
    try:
        with FileLock(str(root)+'.lock',timeout=0):
            generation=json.loads((root/'report.json').read_text())
            expected={shared.case_key(s['name'],2.,r['id']) for s in plan['stages'] for r in plan['cases']}
            if (not generation['complete'] or generation['plan_sha256']!=shared.signature(plan)
                    or len(generation['results'])!=len(expected) or {r['key'] for r in generation['results']}!=expected):
                raise ValueError('Complete matching generation matrix is required before ASR')
            for row in generation['results']:
                verify_artifacts(root,row['artifacts'])
            recognizer, identity=load_asr(plan,root,log)
            previous=root/'comparison.json'
            results=[]
            if previous.exists():
                saved=json.loads(previous.read_text())
                if (saved['plan_sha256']!=shared.signature(plan) or saved['generation_sha256']!=shared.signature(generation)
                        or saved['asr']!=identity):
                    raise ValueError('Comparison inputs or ASR model changed')
                results=saved['results']
                for row in results:verify_artifacts(root,row['artifacts'])
            done={r['key'] for r in results}
            expected_comparisons={f'{s["name"]}/{mode}/{r["id"]}' for s in plan['stages']
                                  for r in plan['original_cases'] for mode in ('whole','sentences')}
            if len(done)!=len(results) or not done<=expected_comparisons:
                raise ValueError('Duplicate or unknown comparison results')
            persist_comparison(root,plan,generation,identity,results)
            for stage in plan['stages']:
                for original in plan['original_cases']:
                    for mode in ('whole','sentences'):
                        key=f'{stage["name"]}/{mode}/{original["id"]}'
                        if stopping or key in done:continue
                        row=assemble(root,plan,stage['name'],original,mode,generation['results'])
                        log(f"[sentence-asr] {len(results)+1}/{plan['total_comparisons']} {key}")
                        segments,_=recognizer.transcribe(str(root/row['audio']),**plan['asr_settings'])
                        segments=[dict(start=s.start,end=s.end,text=s.text,avg_logprob=s.avg_logprob,
                                       no_speech_prob=s.no_speech_prob,
                                       words=[dict(start=w.start,end=w.end,word=w.word,probability=w.probability)
                                              for w in (s.words or [])]) for s in segments]
                        text=' '.join(s['text'].strip() for s in segments)
                        row['asr']=dict(text=text,segments=segments,**transcript_errors(row['text'],text))
                        results.append(row);done.add(key)
                        persist_comparison(root,plan,generation,identity,results)
                        log(f"[sentence-asr] WER={row['asr']['wer']:.4f}; deleted_words={row['asr']['deletion']}")
            report=persist_comparison(root,plan,generation,identity,results)
            if report['complete']:
                shared.package_results(root,results)
                with zipfile.ZipFile(root/'diagnostics-report.zip','a',zipfile.ZIP_DEFLATED) as archive:
                    for name in ('comparison.json','comparison.csv','comparison.md'):
                        archive.write(root/name,name)
                log(f'[sentence-asr] Complete: {root}/comparison.md and diagnostics-*.zip')
            return report
    finally:
        for s,handler in handlers.items():signal.signal(s,handler)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    start=commands.add_parser('start')
    start.add_argument('--run',default=shared.DEFAULT_RUN);start.add_argument('--output',default=DEFAULT_OUTPUT)
    start.add_argument('--job-dir',default=DEFAULT_JOB);start.add_argument('--resume',action='store_true')
    start.add_argument('--step',type=int,default=143);start.add_argument('--cases',nargs='+',default=shared.DEFAULT_CASES)
    start.add_argument('--asr-model',default='large-v3');start.add_argument('--pause-seconds',type=float,default=.2)
    for name in ('status','stop'):
        sub=commands.add_parser(name);sub.add_argument('job',nargs='?',default=DEFAULT_JOB)
    sub=commands.add_parser('summary');sub.add_argument('output',nargs='?',default=DEFAULT_OUTPUT)
    sub=commands.add_parser('_score',help=argparse.SUPPRESS);sub.add_argument('plan')
    args=vars(parser.parse_args());command=args.pop('command')
    if command=='start':
        options={key:args.pop(key) for key in ('step','asr_model','pause_seconds')}
        result=shared.launch(**args,sentence_options=options)
    elif command=='summary':
        print((Path(args['output'])/'comparison.md').read_text());return 0
    elif command=='_score':
        score(json.loads(Path(args['plan']).read_text()));return 0
    else:
        job=Path(args['job']).resolve();result=read_status(job)
        if command=='stop':
            if not result['active']:raise ValueError('Diagnostic job is not active')
            (job/'STOP').touch();result=dict(stop_requested=True,job_dir=str(job))
        else:
            for file,label in [('report.json','generated'),('comparison.json','scored')]:
                p=Path(result['output_dir'])/file
                if p.exists():
                    r=json.loads(p.read_text());result[label]=f"{r['completed_cases']}/{r['total_cases']}"
    print(json.dumps(result,ensure_ascii=False,indent=2));return 0


if __name__=='__main__':
    raise SystemExit(main())

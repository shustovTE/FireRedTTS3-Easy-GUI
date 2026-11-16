"""Codec roundtrip and ground-truth vs generated history on real held-out speech."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import signal
import zipfile

from filelock import FileLock

from . import cfg_diagnostics as shared
from .history_probe import codec_check, history_check, levels, prepare_inputs, reconstruction_metrics
from .sentence_diagnostics import ASR_SETTINGS, load_asr, transcript_errors, verify_artifacts
from .training_data import SpeakerPairs, assert_disjoint, dataset_fingerprint, file_digest, load_manifest
from .training_job import read_status, write_json

SCHEMA = 'native-codec-history-v1'
DEFAULT_OUTPUT = 'training/eval/joint-10-history-diagnostics'
DEFAULT_JOB = 'training/jobs/joint-10-history'
DEFAULT_DATA_RECIPE = 'training/datasets/russian-joint-10/recipe.json'


def select_cases(dialogs, spontaneous, min_long_seconds=15., long_count=2):
    """Longest real targets with distinct same-speaker references; no splicing."""
    cases = []
    for corpus, rows, count, minimum in [('spontaneous', spontaneous, long_count, min_long_seconds),
                                         ('dialogs', dialogs, 1, 0.)]:
        candidates = []
        for target in rows:
            if not minimum <= target['duration'] <= 90:
                continue
            refs = [r for r in rows if r['speaker_id']==target['speaker_id'] and r['language']==target['language']
                    and 2 <= r['duration'] <= 30 and SpeakerPairs.distinct(target, r)
                    and r['text'] != target['text']]
            if refs:
                reference = min(refs, key=lambda r: (abs(r['duration']-6), r['utt_id']))
                candidates.append((target, reference))
        candidates.sort(key=lambda pair: (-pair[0]['duration'], pair[0]['utt_id']))
        chosen, used_speakers = [], set()
        # Prefer different speakers, then fill with remaining longest recordings.
        for diverse in (True, False):
            for pair in candidates:
                if len(chosen) == count:
                    break
                if pair in chosen or (diverse and pair[0]['speaker_id'] in used_speakers):
                    continue
                chosen.append(pair); used_speakers.add(pair[0]['speaker_id'])
        if len(chosen) != count:
            raise ValueError(f'Need {count} real {corpus} targets >= {minimum:g}s with distinct references; '
                             'provide --long-manifest or explicitly lower --min-long-seconds')
        for target, reference in chosen:
            cases.append(dict(id=f'case-{len(cases):02d}', corpus=corpus, target=target, reference=reference))
    return cases


def make_plan(run=shared.DEFAULT_RUN, output=DEFAULT_OUTPUT, step=143, seeds=(1234,1235),
              min_long_seconds=15., long_count=2, long_manifest=None, asr_model='large-v3',
              data_recipe=DEFAULT_DATA_RECIPE):
    from .paths import MODEL_DIR
    from .training_run import prepare_run_data
    run, output = Path(run).resolve(), Path(output).resolve()
    if (type(step) is not int or step <= 0 or not seeds or len(set(seeds)) != len(seeds)
            or any(type(s) is not int or not 0 <= s < 2**32 for s in seeds)
            or type(long_count) is not int or not 1 <= long_count <= 10
            or not 2 <= min_long_seconds <= 90 or not asr_model.strip()):
        raise ValueError('Invalid diagnostic step, seeds, duration, case count or ASR model')
    recipe = json.loads((run/'recipe.json').read_text())
    if recipe['options']['profile'] != 'hybrid' or recipe.get('initial_adapter_fingerprint'):
        raise ValueError('Require the original fresh-base joint run')
    base = MODEL_DIR.resolve()
    for path in (run, base):
        if output==path or output.is_relative_to(path) or path.is_relative_to(output):
            raise ValueError('Diagnostic output must be separate from sources')
    data_recipe = Path(data_recipe).resolve()
    data_config = json.loads(data_recipe.read_text())
    train, dialogs, replay, spontaneous = prepare_run_data(data_config['manifest_path'], recipe['options'],
        recipe['seed'], include_adaptation_validation=True)
    if replay or spontaneous is None:
        raise ValueError('Expected joint data with two holdouts and no replay')
    for key, data in [('train_fingerprint', train), ('validation_fingerprint', dialogs),
                      ('adaptation_validation_fingerprint', spontaneous)]:
        if dataset_fingerprint(data.rows) != recipe[key]:
            raise ValueError(f'Joint data changed since training: {key}')
    long_rows = spontaneous.rows
    if long_manifest:
        long_manifest = str(Path(long_manifest).resolve())
        long_rows = load_manifest(long_manifest)
        assert_disjoint(train.rows, long_rows, scope='recording')
    cases = select_cases(dialogs.rows, long_rows, min_long_seconds, long_count)
    for case in cases:
        for row in (case['target'], case['reference']):
            if output==Path(row['audio']) or output in Path(row['audio']).parents:
                raise ValueError('Output must not contain source audio')
    fingerprint = recipe['options']['expected_base_fingerprint']
    adapter = next((p for p in [run/f'checkpoint-step-{step:06d}', run/f'adapter-step-{step:06d}'] if p.is_dir()), None)
    if adapter is None:
        raise ValueError('Missing selected checkpoint; do not substitute another stage')
    info = shared.adapter_identity(adapter, step, recipe, fingerprint)
    stages = [dict(name='base', step=0, adapter=None),
              dict(name=f'joint-{step}', step=step, adapter=str(adapter), targets=info['targets'],
                   adapter_fingerprint=info['fingerprint'])]
    return dict(schema=SCHEMA, run=str(run), output=str(output), base=str(base), base_fingerprint=fingerprint,
        recipe_sha256=file_digest(run/'recipe.json'), source_step=step, stages=stages, cases=cases, seeds=list(seeds),
        data_recipe=str(data_recipe), data_recipe_sha256=file_digest(data_recipe),
        min_long_seconds=min_long_seconds, long_count=long_count, long_manifest=long_manifest,
        long_manifest_fingerprint=dataset_fingerprint(long_rows), runtime=shared.runtime_identity(),
        settings=dict(shared.SETTINGS), total_cases=len(cases)*(3+len(stages)*len(seeds)*3),
        target_duration_range=[min(c['target']['duration'] for c in cases), max(c['target']['duration'] for c in cases)],
        asr_model=asr_model, asr_settings=dict(ASR_SETTINGS),
        asr_packages={n:importlib.metadata.version(n) for n in ('faster-whisper','ctranslate2','av')})


def verify_plan(plan):
    current = make_plan(plan['run'], plan['output'], plan['source_step'], plan['seeds'],
        plan['min_long_seconds'], plan['long_count'], plan['long_manifest'], plan['asr_model'], plan['data_recipe'])
    if current != plan:
        raise ValueError('History diagnostic inputs/runtime changed; use a new output')


def expected_keys(plan):
    return {f'codec/{c["id"]}/{mode}' for c in plan['cases'] for mode in ('source','codec','codec_with_reference')} | {
        f'{s["name"]}/{c["id"]}/seed-{seed}/{mode}' for s in plan['stages'] for c in plan['cases']
        for seed in plan['seeds'] for mode in ('teacher','self_fixed','free')}


def persist(root, plan, rows):
    complete = len(rows)==plan['total_cases'] and all('audio' in r for r in rows)
    report = dict(schema=SCHEMA, plan_sha256=shared.signature(plan), complete=complete,
        completed_cases=len(rows), total_cases=plan['total_cases'], results=rows, settings=plan['settings'],
        source_step=plan['source_step'], target_duration_range=plan['target_duration_range'],
        note='Inference only; whole utterances. Fixed modes ignore STOP and use the real target length. '
             'Teacher mode conditions on past real patches, never on the current/future patch. '
             'This intervention does not prove that rollout-aware training will work.')
    write_json(root/'report.json', report)
    summary(root, rows, complete, plan)
    return report


def summary(root, rows, complete, plan):
    import csv
    columns = ['stage','id','seed','mode','seconds','finish_reason','end_minus_start_rms_db','peak','pcm_gain','wer']
    lines = ['# Codec and history diagnostics', '', f"Generation complete: {complete}. Rows: {len(rows)}/{plan['total_cases']}.",
             f"Real target durations: {plan['target_duration_range'][0]:.2f}–{plan['target_duration_range'][1]:.2f}s.", '',
             '| Weights | Case | Seed | Mode | Seconds | RMS drift dB | WER |', '|---|---|---:|---|---:|---:|---:|']
    for row in rows:
        wer = f"{row['asr']['wer']:.1%}" if 'asr' in row else 'pending'
        lines.append(f"| {row['stage']} | {row['id']} | {row['seed']} | {row['mode']} | {row['seconds']:.2f} | "
                     f"{row['end_minus_start_rms_db']:+.2f} | {wer} |")
    lines += ['', 'RMS uses non-overlapping head/tail windows (up to 10 seconds each).',
        'Codec reconstruction: inspect active-window gain relative to source, tail_gain_minus_head_gain_db and spectral_convergence.',
        'teacher and self_fixed have the same target horizon; both record but ignore STOP. free respects native STOP/400-patch cap.',
        'Compare teacher vs self_fixed within weights and seed, then free; same reference, text, noise schedule and decoder.',
        'One listening gain per case across every variant; raw FLOAT WAV retained locally.',
        'ASR may be wrong; source ASR establishes a recognition control. No automatic winner or perceptual approval.']
    (root/'summary.md').write_text('\n'.join(lines)+'\n')
    temporary=root/'summary.csv.tmp'
    with temporary.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=columns,extrasaction='ignore'); writer.writeheader()
        writer.writerows({**row,'wer':row.get('asr',{}).get('wer')} for row in rows)
    temporary.replace(root/'summary.csv')


def write_row(root, case, key, samples, rate, *, stage, mode, seed=None, **metrics):
    import soundfile as sf
    path=root/'audio'/f'{key}.float.wav'; path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp')
    sf.write(temporary,samples,rate,format='WAV',subtype='FLOAT'); temporary.replace(path)
    return dict(key=key,id=case['id'],stage=stage,mode=mode,seed=seed,text=case['target']['text'],
        target_sha256=case['target']['audio_sha256'],reference_sha256=case['reference']['audio_sha256'],
        **levels(samples,rate),**metrics,float_audio=str(path.relative_to(root)),
        artifacts={str(path.relative_to(root)):file_digest(path)})


def listening_copies(root, rows):
    import soundfile as sf
    for row in rows:
        gain=min(1.,.98/max(1e-12,max(r['peak'] for r in rows if r['id']==row['id'])))
        if 'audio' in row:
            if row['pcm_gain']!=gain:
                raise ValueError('Listening gain differs from the completed comparison')
            continue
        wave,rate=sf.read(root/row['float_audio'],dtype='float32')
        path=root/row['float_audio'].replace('.float.wav','.wav')
        temporary=path.with_suffix('.tmp')
        sf.write(temporary,wave*gain,rate,format='WAV',subtype='PCM_16'); temporary.replace(path)
        name=str(path.relative_to(root))
        row.update(audio=name,pcm_gain=gain)
        row['artifacts'][name]=file_digest(path)


def asr_input(root, row):
    """Avoid quantizing all ASR inputs against a potentially runaway loud variant."""
    import numpy as np
    import soundfile as sf
    import torch
    import torchaudio
    wave, rate = sf.read(root/row['float_audio'], dtype='float32')
    wave = torchaudio.functional.resample(torch.from_numpy(wave), rate, 16000).numpy()
    if not np.isfinite(wave).all():
        raise ValueError('Nonfinite ASR input')
    gain = min(1., .98/max(float(np.abs(wave).max()), 1e-12))
    return wave*gain, gain


def execute(plan, *, resume=False, manager=None, log=print):
    import torch
    from .model_manager import ModelManager
    from .training_data import model_identity
    from .training_objective import adapter_targets, validate_adapter_targets
    verify_plan(plan); shared.require_cuda()
    log('[history] Verifying original base files')
    if model_identity(plan['base'])!=plan['base_fingerprint']:
        raise ValueError('Base weights changed')
    root=Path(plan['output']); root.parent.mkdir(parents=True,exist_ok=True)
    manager=manager or ModelManager()
    stopping=False
    def stop(_s,_f):
        nonlocal stopping
        stopping=True
        log('[history] Stop requested; finish the current three-mode comparison and save')
    handlers={s:signal.signal(s,stop) for s in (signal.SIGINT,signal.SIGTERM)}
    try:
        with FileLock(str(root)+'.lock',timeout=0):
            if resume:
                if json.loads((root/'plan.json').read_text())!=plan:
                    raise ValueError('Resume plan differs')
                saved=json.loads((root/'report.json').read_text())
                if saved['plan_sha256']!=shared.signature(plan):raise ValueError('Report plan differs')
                rows=saved['results']
                for r in rows:verify_artifacts(root,r['artifacts'])
            else:
                root.mkdir(exist_ok=False); write_json(root/'plan.json',plan); rows=[]
                persist(root,plan,rows)
            done={r['key'] for r in rows}
            if len(done)!=len(rows) or not done <= expected_keys(plan):
                raise ValueError('Duplicate or unknown completed results')
            for stage in plan['stages']:
                if stopping:break
                pending=[c for c in plan['cases'] if any(
                    f'{stage["name"]}/{c["id"]}/seed-{seed}/{mode}' not in done
                    for seed in plan['seeds'] for mode in ('teacher','self_fixed','free'))
                    or (stage['name']=='base' and any(f'codec/{c["id"]}/{m}' not in done
                                                    for m in ('source','codec','codec_with_reference')))]
                if not pending:continue
                manager.unload(log)
                tts=manager.load_base(attention_backend='sdpa',torch_compile=False,log=log)
                try:
                    if stage['adapter']:
                        validate_adapter_targets(tts.tts_core,stage['targets'],adapter_targets(tts.tts_core,'hybrid'))
                        manager._attach_adapter(tts,stage['adapter'],log)
                    for module in (tts.tts_core,tts.redae,tts.spk_extractor):
                        module.eval(); module.requires_grad_(False)
                    for case in pending:
                        if stopping:break
                        log(f"[history] {stage['name']} {case['id']} target={case['target']['duration']:.2f}s")
                        inputs=prepare_inputs(tts,case)
                        padding=dict(target_left_padding_samples=inputs['target_pad'],
                            prompt_left_padding_samples=inputs['prompt_pad'],prompt_samples=inputs['prompt_samples'],
                            source_samples=inputs['source_samples'],text_tokens=inputs['tokens'].numel(),effective_text=inputs['effective_text'])
                        if stage['name']=='base':
                            keys={f'codec/{case["id"]}/{m}' for m in ('source','codec','codec_with_reference')}
                            if not keys <= done:
                                codecs=codec_check(tts,inputs)
                                for mode,wave in codecs.items():
                                    key=f'codec/{case["id"]}/{mode}'
                                    if key in done:continue
                                    reconstruction=reconstruction_metrics(codecs['source'],wave,inputs['rate']) if mode!='source' else None
                                    rows.append(write_row(root,case,key,wave,inputs['rate'],stage='codec',mode=mode,
                                        finish_reason='real_recording',reconstruction=reconstruction,**padding))
                                    done.add(key)
                                persist(root,plan,rows)
                        for seed in plan['seeds']:
                            if stopping:break
                            prefix=f'{stage["name"]}/{case["id"]}/seed-{seed}'
                            if all(prefix+'/'+m in done for m in ('teacher','self_fixed','free')):continue
                            log(f'[history] {prefix}: native free, fixed self-history, fixed real history')
                            results=history_check(tts,inputs,seed,plan['settings'])
                            for result in results:
                                mode=result.pop('mode'); wave=result.pop('samples'); key=prefix+'/'+mode
                                if key in done:continue
                                rows.append(write_row(root,case,key,wave,inputs['rate'],stage=stage['name'],
                                    mode=mode,seed=seed,**padding,**result)); done.add(key)
                            persist(root,plan,rows)
                            log(f"[history] Saved {len(rows)}/{plan['total_cases']} outputs")
                finally:
                    tts=None; manager.unload(log)
            if len(rows)==plan['total_cases']:
                listening_copies(root,rows)
            report=persist(root,plan,rows)
            log('[history] Generation complete; CPU ASR follows' if report['complete'] else '[history] Paused; resume with the same plan')
            return report
    finally:
        manager.unload(log)
        for s,h in handlers.items():signal.signal(s,h)


def score(plan, log=print):
    verify_plan(plan); root=Path(plan['output']); stopping=False
    def stop(_s,_f):
        nonlocal stopping
        stopping=True
        log('[history-asr] Stop requested; finish this WAV and save')
    handlers={s:signal.signal(s,stop) for s in (signal.SIGINT,signal.SIGTERM)}
    try:
        with FileLock(str(root)+'.lock',timeout=0):
            generation=json.loads((root/'report.json').read_text())
            if (not generation['complete'] or generation['plan_sha256']!=shared.signature(plan)
                    or len(generation['results'])!=plan['total_cases']
                    or {r['key'] for r in generation['results']}!=expected_keys(plan)):
                raise ValueError('Complete matching generation report required')
            for row in generation['results']:verify_artifacts(root,row['artifacts'])
            recognizer,identity=load_asr(plan,root,log)
            path=root/'asr_report.json'
            report=dict(plan_sha256=shared.signature(plan),generation_sha256=shared.signature(generation),asr=identity,
                        complete=False,completed_cases=0,total_cases=plan['total_cases'],results=[])
            if path.exists():
                saved=json.loads(path.read_text())
                if any(saved[k]!=report[k] for k in ('plan_sha256','generation_sha256','asr')):
                    raise ValueError('ASR inputs changed')
                report=saved
            done={r['key'] for r in report['results']}
            if len(done)!=len(report['results']) or not done<=expected_keys(plan):
                raise ValueError('Duplicate or unknown ASR results')
            for row in report['results']:verify_artifacts(root,row['artifacts'])
            for row in generation['results']:
                if stopping:break
                if row['key'] in done:continue
                log(f"[history-asr] {len(done)+1}/{plan['total_cases']} {row['key']}")
                audio, asr_gain=asr_input(root,row)
                segments,_=recognizer.transcribe(audio,**plan['asr_settings'])
                segments=[dict(start=s.start,end=s.end,text=s.text,
                    words=[dict(start=w.start,end=w.end,word=w.word,probability=w.probability) for w in (s.words or [])])
                    for s in segments]
                transcript=' '.join(s['text'].strip() for s in segments)
                result=dict(row,asr=dict(text=transcript,segments=segments,input_gain=asr_gain,input_sample_rate=16000,
                    input_source='float_audio',**transcript_errors(row['text'],transcript)))
                report['results'].append(result); done.add(row['key'])
                report.update(completed_cases=len(done),complete=len(done)==plan['total_cases'])
                write_json(path,report); summary(root,report['results'],True,plan)
            summary(root,report['results'],True,plan)
            if report['complete']:
                shared.package_results(root,report['results'])
                with zipfile.ZipFile(root/'diagnostics-report.zip','a',zipfile.ZIP_DEFLATED) as archive:
                    archive.write(path,path.name)
                log(f'[history] Complete: {root}/diagnostics-*.zip')
            return report
    finally:
        for s,h in handlers.items():signal.signal(s,h)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    start=commands.add_parser('start')
    start.add_argument('--run',default=shared.DEFAULT_RUN); start.add_argument('--output',default=DEFAULT_OUTPUT)
    start.add_argument('--job-dir',default=DEFAULT_JOB); start.add_argument('--resume',action='store_true')
    start.add_argument('--step',type=int,default=143); start.add_argument('--seeds',type=int,nargs='+',default=[1234,1235])
    start.add_argument('--min-long-seconds',type=float,default=15.); start.add_argument('--long-count',type=int,default=2)
    start.add_argument('--long-manifest'); start.add_argument('--asr-model',default='large-v3')
    start.add_argument('--data-recipe',default=DEFAULT_DATA_RECIPE)
    for name in ('status','stop'):
        sub=commands.add_parser(name); sub.add_argument('job',nargs='?',default=DEFAULT_JOB)
    sub=commands.add_parser('summary'); sub.add_argument('output',nargs='?',default=DEFAULT_OUTPUT)
    sub=commands.add_parser('_run',help=argparse.SUPPRESS); sub.add_argument('plan'); sub.add_argument('--resume',action='store_true')
    sub=commands.add_parser('_score',help=argparse.SUPPRESS); sub.add_argument('plan')
    args=vars(parser.parse_args()); command=args.pop('command')
    if command=='start':
        options={k:args.pop(k) for k in ('step','seeds','min_long_seconds','long_count','long_manifest','asr_model','data_recipe')}
        result=shared.launch(**args,history_options=options)
    elif command in ('_run','_score'):
        plan=json.loads(Path(args.pop('plan')).read_text())
        (execute if command=='_run' else score)(plan,**args); return 0
    elif command=='summary':
        print((Path(args['output'])/'summary.md').read_text()); return 0
    else:
        job=Path(args['job']).resolve(); result=read_status(job)
        if command=='stop':
            if not result['active']:raise ValueError('Diagnostic job is not active')
            (job/'STOP').touch(); result=dict(stop_requested=True,job_dir=str(job))
        else:
            for name,label in [('report.json','generated'),('asr_report.json','scored')]:
                path=Path(result['output_dir'])/name
                if path.exists():
                    saved=json.loads(path.read_text()); result[label]=f"{saved['completed_cases']}/{saved['total_cases']}"
    print(json.dumps(result,ensure_ascii=False,indent=2)); return 0


if __name__=='__main__':
    raise SystemExit(main())

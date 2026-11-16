"""Read-only native CFG/checkpoint matrix for the joint-10 acoustic regression."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import zipfile

from filelock import FileLock

from .training_data import file_digest
from .training_job import now, read_status, run_commands, write_json
from .training_monitor import SETTINGS, failures, generate_case, load_benchmark

DEFAULT_RUN = 'training/outputs/russian-joint-base-10'
DEFAULT_OUTPUT = 'training/eval/joint-10-cfg-diagnostics'
DEFAULT_JOB = 'training/jobs/joint-10-cfg'
DEFAULT_CASES = ['cities_seed1235', 'timofey_seed1235']
SCHEMA = 'native-cfg-diagnostics-v1'


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def runtime_identity():
    import importlib.metadata
    import torch
    root = Path(__file__).resolve().parent.parent
    files = sorted((root/'fireredtts3_upstream').rglob('*.py'))
    files += [root/'fireredtts3_local'/name for name in
              ('cfg_diagnostics.py', 'component_diagnostics.py', 'sentence_diagnostics.py',
               'native_comparison.py', 'training_parity.py', 'training_batch.py',
               'training_monitor.py', 'model_manager.py', 'hybrid.py',
               'training_data.py', 'training_run.py', 'training_objective.py')]
    return dict(code={str(p.relative_to(root)): file_digest(p) for p in files},
                packages={name: importlib.metadata.version(name) for name in
                          ('torch', 'torchaudio', 'transformers', 'peft', 'numpy', 'soundfile')},
                cuda=torch.version.cuda)


def adapter_identity(path, step, recipe, base_fingerprint):
    from .training_run import initial_adapter_info
    metadata = json.loads((path/'firered_lora_metadata.json').read_text())
    if metadata.get('step') != step or metadata.get('recipe') != recipe:
        raise ValueError(f'Checkpoint step/recipe differs from the selected training run: {path}')
    info = initial_adapter_info(path, recipe['rank'], recipe['alpha'], recipe['dropout'], 'hybrid')
    if info['base_fingerprint'] != base_fingerprint:
        raise ValueError(f'Checkpoint base fingerprint mismatch: {path}')
    return info


def make_plan(run=DEFAULT_RUN, output=DEFAULT_OUTPUT, steps=(50,143,286),
              cfgs=(0.,1.,2.), cases=DEFAULT_CASES):
    from .paths import MODEL_DIR
    run, output, base = Path(run).resolve(), Path(output).resolve(), MODEL_DIR.resolve()
    for source in (run, base):
        if output == source or output.is_relative_to(source) or source.is_relative_to(output):
            raise ValueError('Diagnostic output must be separate from training and original weights')
    if (not steps or len(set(steps)) != len(steps) or any(type(n) is not int or n <= 0 for n in steps)
            or not cfgs or len(set(cfgs)) != len(cfgs) or any(not math.isfinite(c) or c < 0 for c in cfgs)
            or not cases or len(set(cases)) != len(cases)):
        raise ValueError('Require unique positive checkpoint steps, nonnegative finite CFG values and case IDs')
    recipe = json.loads((run/'recipe.json').read_text())
    if recipe['options']['profile'] != 'hybrid' or recipe.get('initial_adapter_fingerprint'):
        raise ValueError('Expected the fresh-base joint hybrid run')
    base_fingerprint = recipe['options']['expected_base_fingerprint']
    if not base_fingerprint:
        raise ValueError('Training run must pin the original base')
    rows, benchmark_fingerprint = load_benchmark(recipe['options']['generation_benchmark'])
    if benchmark_fingerprint != recipe['generation_benchmark_fingerprint']:
        raise ValueError('Benchmark text/reference/settings changed since training')
    indexed = {r['id']: r for r in rows}
    if set(cases)-indexed.keys():
        raise ValueError(f'Missing benchmark cases: {sorted(set(cases)-indexed.keys())}')
    rows = [indexed[key] for key in cases]
    if any(output == Path(r['reference_audio']) or output in Path(r['reference_audio']).parents for r in rows):
        raise ValueError('Output must not contain source reference audio')
    stages = [dict(name='base', step=0, adapter=None)]
    for step in steps:
        choices = [run/f'checkpoint-step-{step:06d}', run/f'adapter-step-{step:06d}']
        adapter = next((p for p in choices if p.is_dir()), None)
        if adapter is None:
            raise ValueError(f'Missing saved checkpoint for step {step}; do not substitute best-validation')
        info = adapter_identity(adapter, step, recipe, base_fingerprint)
        stages.append(dict(name=f'step-{step:06d}', step=step, adapter=str(adapter),
                           adapter_fingerprint=info['fingerprint'], targets=info['targets']))
    previous = {}
    summaries = [json.loads(line) for line in (run/'generation_metrics.jsonl').read_text().splitlines() if line.strip()]
    for summary in summaries:
        if summary['step'] not in {s['step'] for s in stages}:
            continue
        path = Path(summary['report']).resolve()
        report = json.loads(path.read_text())
        if (not report.get('complete') or report['fingerprint'] != benchmark_fingerprint
                or report['settings'] != SETTINGS or report['step'] != summary['step']):
            raise ValueError('Previous generation report has inconsistent provenance/settings')
        previous[str(summary['step'])] = dict(path=str(path), sha256=file_digest(path))
    if '0' not in previous:
        raise ValueError('Original step-0 generation report is required for reproduction checks')
    return dict(schema=SCHEMA, run=str(run), output=str(output), base=str(base),
                base_fingerprint=base_fingerprint, recipe_sha256=file_digest(run/'recipe.json'),
                benchmark_fingerprint=benchmark_fingerprint, cases=rows, stages=stages,
                cfgs=[float(c) for c in cfgs], settings=SETTINGS, previous_reports=previous,
                runtime=runtime_identity(), total_cases=len(stages)*len(cfgs)*len(rows))


def verify_plan(plan):
    if plan['schema'] == SCHEMA:
        current = make_plan(plan['run'], plan['output'], [s['step'] for s in plan['stages'][1:]],
                            plan['cfgs'], [r['id'] for r in plan['cases']])
    elif plan['schema'] == 'native-sentence-diagnostics-v1':
        from .sentence_diagnostics import make_plan as sentence_plan
        current = sentence_plan(plan['run'], plan['output'], plan['source_step'],
                                [r['id'] for r in plan['original_cases']],
                                plan['asr_model'], plan['pause_seconds'])
    else:
        from .component_diagnostics import SCHEMA as COMPONENT_SCHEMA, SPLIT_SCHEMA, make_plan as component_plan
        if plan['schema'] not in (COMPONENT_SCHEMA, SPLIT_SCHEMA):
            raise ValueError('Unknown diagnostic schema')
        current = component_plan(plan['run'], plan['output'], plan['ablation_step'],
                                 [r['id'] for r in plan['cases']], split_acoustic=plan['schema']==SPLIT_SCHEMA)
    if current != plan:
        raise ValueError('Weights, benchmark, training reports or runtime changed; use a new diagnostic output')


def case_key(stage, cfg, case):
    return f'{stage}/cfg-{cfg:g}/{case}'


def require_cuda():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('Run this diagnostic on the CUDA host with the original base and checkpoints')


def latent_summary(result):
    # The first PatchEncoder input is the reference, not generated speech.
    latents = result['latent_levels'][1:]
    n = min(len(latents), max(1, round(10*len(latents)/result['seconds'])))
    def average(values):
        return sum(values)/len(values) if values else None
    first = average([r['rms'] for r in latents[:n]])
    last = average([r['rms'] for r in latents[-n:]])
    flow = result['flow_by_patch']
    return dict(first_latent_rms=first, last_latent_rms=last,
                latent_rms_ratio=last/max(first,1e-12) if first is not None else None,
                first_guided_flow_rms=average([r['guided_rms'] for r in flow[:n]]),
                last_guided_flow_rms=average([r['guided_rms'] for r in flow[-n:]]))


def annotate(results, plan):
    """Compare each adapter only with the matching base CFG/text/seed."""
    baseline = {(r['cfg'], r['id']): r for r in results if r['stage']=='base'}
    old = {}
    for step, source in plan['previous_reports'].items():
        report = json.loads(Path(source['path']).read_text())
        old.update({(int(step), r['id']): r for r in report['results']})
    for row in results:
        reference = baseline.get((row['cfg'], row['id']))
        row['failures'] = failures(row, reference)
        if 'training_parity' in row:
            row['training_parity_passed'] = row['training_parity']['passed']
            if not row['training_parity_passed']:
                row['failures'].append('training_forward_mismatch')
        row['rms_drift_delta_vs_base_same_cfg'] = (
            row['end_minus_start_rms_db']-reference['end_minus_start_rms_db'] if reference else None)
        # Partial components intentionally differ from their source checkpoint.
        # Only the base and complete-checkpoint controls reproduce old results.
        previous_step = row.get('comparison_step', row['step'])
        previous_id = row.get('comparison_id', row['id'])
        previous = old.get((previous_step, previous_id)) if row['cfg']==2. else None
        row['previous_monitor'] = None if previous is None else dict(
            seconds_delta=row['seconds']-previous['seconds'],
            rms_drift_delta_db=row['end_minus_start_rms_db']-previous['end_minus_start_rms_db'],
            finish_matches=row['finish_reason']==previous['finish_reason'],
            reproduction_review_required=(row['finish_reason']!=previous['finish_reason']
                or abs(row['seconds']-previous['seconds'])>.161
                or abs(row['end_minus_start_rms_db']-previous['end_minus_start_rms_db'])>.25))


def persist(output, plan, results):
    annotate(results, plan)
    report = dict(schema=plan['schema'], plan_sha256=signature(plan), complete=len(results)==plan['total_cases'],
                  completed_cases=len(results), total_cases=plan['total_cases'], results=results,
                  note='Native inference only. No weights updated. No ASR or speaker-similarity scoring; '
                       'duration, RMS and engineering checks do not certify pronunciation or timbre.')
    if 'ablation_step' in plan:
        report['ablation_step'] = plan['ablation_step']
        report['component_definitions'] = plan['component_definitions']
        report['reproduction_controls_complete'] = all(any(
            r['stage']==stage and r['id']==case['id'] and r.get('previous_monitor') is not None
            for r in results) for stage in ('base','all') for case in plan['cases'])
        report['reproduction_review_required'] = any(
            r.get('previous_monitor') and r['previous_monitor']['reproduction_review_required'] for r in results)
    if plan.get('training_parity'):
        report['parity_review_required'] = any(not r['training_parity']['passed'] for r in results)
    write_json(output/'report.json', report)
    columns = ('stage','step','active_components','cfg','id','seed','seconds','finish_reason','end_minus_start_rms_db',
               'rms_drift_delta_vs_base_same_cfg','fraction_at_or_above_full_scale','peak',
               'first_latent_rms','last_latent_rms','latent_rms_ratio','pcm_gain','training_parity_passed','failures','audio')
    temporary = output/'summary.csv.tmp'
    with temporary.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction='ignore')
        writer.writeheader(); writer.writerows(results)
    temporary.replace(output/'summary.csv')
    title = 'Native component diagnostics' if 'ablation_step' in plan else 'Native CFG diagnostics'
    lines = ['# '+title, '', f"Completed {len(results)}/{plan['total_cases']} cases.", '',
             '| Weights | CFG | Case | Seconds | Finish | RMS drift dB | Δ vs base, same CFG | Outside ±1 % | Checks |',
             '|---|---:|---|---:|---|---:|---:|---:|---|']
    for r in results:
        delta = r['rms_drift_delta_vs_base_same_cfg']
        lines.append(f"| {r['stage']} | {r['cfg']:g} | {r['id']} | {r['seconds']:.2f} | {r['finish_reason']} | "
                     f"{r['end_minus_start_rms_db']:+.2f} | {delta:+.2f} | "
                     f"{100*r['fraction_at_or_above_full_scale']:.3f} | {', '.join(r['failures']) or 'no flags'} |"
                     if delta is not None else f"| {r['stage']} | {r['cfg']:g} | {r['id']} | pending base |")
    lines += ['', 'CFG 0 uses the conditional branch alone. CFG 2 = 3 × conditional − 2 × unconditional.',
              'Each WAV uses its case seed, STOP 0.5, 10 ODE steps, and no sentence splitting.',
              'Listening copies use fixed gain only when needed; inspect raw float WAV/metrics for level comparisons.',
              'Fewer seconds or no flags do not prove that all words were spoken. ASR is not run.',
              'Check previous_monitor in report.json: materially different CFG=2 reproduction needs investigation.']
    if 'ablation_step' in plan:
        lines += ['', f"All trained components come from checkpoint {plan['ablation_step']}.",
                  *[f'{name}: {definition}' for name, definition in plan['component_definitions'].items()],
                  'Unselected components use original base weights. RedAE, speaker extraction/projections and PatchEncoder are unchanged.',
                  'base and all are reproduction controls; partial variants are not expected to match the full checkpoint.',
                  'If either control differs materially from its prior report, investigate reproducibility before attributing a cause.',
                  'Mixed components were not trained independently. Their interactions can change generation; no automatic culprit or winner is selected.']
    if plan.get('training_parity'):
        lines += ['', f"Forward parity review required: {report['parity_review_required']}.",
                  'Each case replays its first/last four generated patches through the training forward on identical history at flow t=0.',
                  'Inspect training_parity.checks per case. BF16 full-sequence vs cache uses diagnostic tolerances, not exact equality.',
                  'Parity does not validate the training objective, gradients or free-running robustness.']
    (output/'summary.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return report


def package_results(output, results, chunk_bytes=24*1024*1024):
    """Small sharing archives; no reference voice, transcript, weights or token."""
    reports = [output/name for name in ('report.json','summary.csv','summary.md')]
    with zipfile.ZipFile(output/'diagnostics-report.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for path in reports:
            archive.write(path,path.relative_to(output))
    groups, group, size = [], [], 0
    for result in results:
        path=output/result['audio']
        if path.stat().st_size > chunk_bytes:
            raise ValueError('Unexpected oversized listening WAV')
        if group and size+path.stat().st_size > chunk_bytes:
            groups.append(group);group=[];size=0
        group.append(path);size+=path.stat().st_size
    if group:groups.append(group)
    for index, group in enumerate(groups,1):
        with zipfile.ZipFile(output/f'diagnostics-audio-{index:02d}.zip','w',zipfile.ZIP_DEFLATED) as archive:
            for path in group:
                archive.write(path,path.relative_to(output))


def execute(plan, *, resume=False, manager=None, log=print):
    import soundfile as sf
    import torch
    from .model_manager import ModelManager
    from .training_data import model_identity
    from .training_objective import adapter_targets, validate_adapter_targets
    verify_plan(plan)
    require_cuda()
    log('[cfg-diagnostics] Verifying original base fingerprint before loading GPU models')
    if model_identity(plan['base']) != plan['base_fingerprint']:
        raise ValueError('Original base differs from the training base')
    output=Path(plan['output']);output.parent.mkdir(parents=True,exist_ok=True)
    stopping=False
    def stop(_signal,_frame):
        nonlocal stopping
        stopping=True
        log('[cfg-diagnostics] Stop requested; finish this case and preserve completed results')
    handlers={sig:signal.signal(sig,stop) for sig in (signal.SIGINT,signal.SIGTERM)}
    manager=manager or ModelManager()
    try:
        with FileLock(str(output)+'.lock',timeout=0):
            if resume:
                if json.loads((output/'plan.json').read_text()) != plan:
                    raise ValueError('Resume plan differs from the saved matrix')
                saved=json.loads((output/'report.json').read_text())
                if saved['plan_sha256'] != signature(plan):
                    raise ValueError('Resume report belongs to different inputs')
                results=saved['results']
                for row in results:
                    for name, digest in row['artifacts'].items():
                        path=(output/name).resolve()
                        if not path.is_relative_to(output) or file_digest(path) != digest:
                            raise ValueError(f'Completed diagnostic audio changed: {name}')
            else:
                output.mkdir(exist_ok=False)
                write_json(output/'plan.json',plan)
                results=[]
                persist(output,plan,results)
            done={r['key'] for r in results}
            expected={case_key(s['name'],cfg,r['id']) for s in plan['stages'] for cfg in plan['cfgs'] for r in plan['cases']}
            if len(done)!=len(results) or not done<=expected:
                raise ValueError('Unexpected or duplicate completed case keys')
            for stage in plan['stages']:
                pending=[(cfg,row) for cfg in plan['cfgs'] for row in plan['cases']
                         if case_key(stage['name'],cfg,row['id']) not in done]
                if not pending or stopping:
                    continue
                manager.unload(log)
                tts=manager.load_base(attention_backend='sdpa',torch_compile=False,log=log)
                try:
                    if stage['adapter']:
                        validate_adapter_targets(tts.tts_core,stage['targets'],adapter_targets(tts.tts_core,'hybrid'))
                        manager._attach_adapter(tts,stage['adapter'],log)
                    component_audit = None
                    if 'active_components' in stage:
                        from .component_diagnostics import select_components
                        if stage['adapter']:
                            component_audit = select_components(tts.tts_core,stage['active_components'])
                        elif stage['active_components']:
                            raise ValueError('Trained components require an adapter')
                        else:
                            component_audit = dict(active_components=[], original_base=True)
                    for module in (tts.tts_core,tts.redae,tts.spk_extractor):
                        module.eval();module.requires_grad_(False)
                    for cfg,row in pending:
                        if stopping:break
                        key=case_key(stage['name'],cfg,row['id'])
                        log(f"[cfg-diagnostics] {len(results)+1}/{plan['total_cases']} {key}")
                        if file_digest(row['reference_audio']) != row['reference_sha256']:
                            raise ValueError('Reference audio changed during diagnostic')
                        settings={**plan['settings'],'inference_cfg':cfg}
                        if plan.get('training_parity'):
                            from .training_parity import generate_with_parity
                            samples,rate,metrics=generate_with_parity(tts,row,settings,generate_case)
                            log(f"[training-parity] {key}: passed={metrics['training_parity']['passed']}")
                        else:
                            samples,rate,metrics=generate_case(tts,row,settings,trace_flow=True)
                        # Indexed filenames keep arbitrary benchmark IDs out of paths.
                        name=f"case-{plan['cases'].index(row):02d}"
                        folder=output/stage['name']/f'cfg-{cfg:g}';folder.mkdir(parents=True,exist_ok=True)
                        gain=min(1.,.98/max(metrics['peak'],1e-12))
                        paths=[folder/f'{name}.float.wav',folder/f'{name}.wav']
                        for path,values,subtype in zip(paths,(samples,samples*gain),('FLOAT','PCM_16')):
                            temporary=path.with_suffix('.tmp')
                            sf.write(temporary,values,rate,format='WAV',subtype=subtype)
                            temporary.replace(path)
                        result=dict(key=key,stage=stage['name'],step=stage['step'],cfg=cfg,id=row['id'],seed=row['seed'],
                            text=row['text'],settings={**plan['settings'],'inference_cfg':cfg},**metrics,
                            **latent_summary(metrics),pcm_gain=gain,
                            audio=str(paths[1].relative_to(output)),float_audio=str(paths[0].relative_to(output)),
                            artifacts={str(p.relative_to(output)):file_digest(p) for p in paths})
                        for field in ('source_case_id', 'mode', 'sentence_index', 'comparison_id'):
                            if field in row:
                                result[field] = row[field]
                        if component_audit is not None:
                            result.update(active_components=stage['active_components'],component_audit=component_audit,
                                          comparison_step=stage['comparison_step'])
                        results.append(result);done.add(key)
                        persist(output,plan,results)
                        log(f"[cfg-diagnostics] {key}: {metrics['seconds']:.2f}s; {metrics['finish_reason']}; "
                            f"RMS drift={metrics['end_minus_start_rms_db']:+.2f}dB; "
                            f"outside ±1={100*metrics['fraction_at_or_above_full_scale']:.3f}%")
                finally:
                    tts=None
                    manager.unload(log)
            report=persist(output,plan,results)
            if report['complete']:
                if 'original_cases' in plan:
                    log('[cfg-diagnostics] Generation complete; sentence assembly and CPU ASR follow')
                else:
                    package_results(output,results)
                    log(f'[cfg-diagnostics] Complete: {output}/summary.md and diagnostics-*.zip')
            else:
                log(f"[cfg-diagnostics] Paused after {len(results)}/{plan['total_cases']} cases; resume with the same inputs")
            return report
    finally:
        manager.unload(log)
        for sig,handler in handlers.items():signal.signal(sig,handler)


def launch(run=DEFAULT_RUN, output=DEFAULT_OUTPUT, job_dir=DEFAULT_JOB, *, resume=False,
           steps=(50,143,286), cfgs=(0.,1.,2.), cases=DEFAULT_CASES, component_step=None, split_acoustic=False,
           sentence_options=None):
    import fcntl
    run,output,job=Path(run).resolve(),Path(output).resolve(),Path(job_dir).resolve()
    if job.exists():raise ValueError('Use a new job directory, including when resuming')
    for source in (run,output):
        if job==source or job.is_relative_to(source) or source.is_relative_to(job):
            raise ValueError('Job, diagnostic output and source run must be separate directories')
    if output.exists()!=resume:
        raise ValueError('Output exists: use --resume; for a new diagnostic use a new output directory')
    # Share the existing training-output lock: training cannot rewrite these
    # checkpoints while the diagnostic supervisor/child holds the inherited FD.
    locks=Path('.runtime/training-job-locks').resolve();locks.mkdir(parents=True,exist_ok=True)
    lock_path=locks/(hashlib.sha256(str(run).encode()).hexdigest()+'.lock')
    with lock_path.open('a+') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as exc:raise ValueError('Training or a diagnostic already owns this source run') from exc
        if sentence_options is not None:
            if component_step is not None or split_acoustic:
                raise ValueError('Sentence and component diagnostic modes are mutually exclusive')
            from .sentence_diagnostics import make_plan as sentence_plan
            plan=sentence_plan(run,output,cases=cases,**sentence_options)
        elif component_step is None:
            plan=make_plan(run,output,steps,cfgs,cases)
        else:
            from .component_diagnostics import make_plan as component_plan
            plan=component_plan(run,output,component_step,cases,split_acoustic=split_acoustic)
        if resume and json.loads((output/'plan.json').read_text())!=plan:
            raise ValueError('Resume inputs or runtime changed; use a new output')
        job.mkdir(parents=True,exist_ok=False)
        lock.seek(0);lock.truncate();lock.write(str(job));lock.flush()
        write_json(job/'plan.json',plan)
        state=dict(status='starting',created_at=now(),cwd=str(Path.cwd()),python=sys.executable,
                   output_dir=str(output),source_run=str(run),lock_path=str(lock_path),
                   log=str(job/'job.log'),resume=resume,total_cases=plan['total_cases'])
        write_json(job/'status.json',state)
        try:
            with (job/'job.log').open('ab',buffering=0) as stream:
                process=subprocess.Popen([sys.executable,'-u','-m',__package__+'.cfg_diagnostics',
                    '_worker',str(job),str(lock.fileno())],stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,
                    start_new_session=True,close_fds=True,pass_fds=(lock.fileno(),))
        except BaseException:
            state.update(status='failed',finished_at=now(),error='Could not launch diagnostic worker')
            write_json(job/'status.json',state)
            raise
    return dict(pid=process.pid,job_dir=str(job),log=str(job/'job.log'),output=str(output),cases=plan['total_cases'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    start=commands.add_parser('start',help='Run the matrix in a detached background job')
    start.add_argument('--run',default=DEFAULT_RUN);start.add_argument('--output',default=DEFAULT_OUTPUT)
    start.add_argument('--job-dir',default=DEFAULT_JOB);start.add_argument('--resume',action='store_true')
    start.add_argument('--steps',type=int,nargs='+',default=[50,143,286])
    start.add_argument('--cfgs',type=float,nargs='+',default=[0.,1.,2.])
    start.add_argument('--cases',nargs='+',default=DEFAULT_CASES)
    for name in ('status','stop'):
        sub=commands.add_parser(name);sub.add_argument('job',nargs='?',default=DEFAULT_JOB)
    sub=commands.add_parser('summary');sub.add_argument('output',nargs='?',default=DEFAULT_OUTPUT)
    sub=commands.add_parser('_worker',help=argparse.SUPPRESS);sub.add_argument('job');sub.add_argument('lock_fd',type=int)
    sub=commands.add_parser('_run',help=argparse.SUPPRESS);sub.add_argument('plan');sub.add_argument('--resume',action='store_true')
    args=vars(parser.parse_args());command=args.pop('command')
    if command=='start':result=launch(**args)
    elif command in {'status','stop'}:
        job=Path(args['job']).resolve();result=read_status(job)
        if command=='stop':
            if not result['active']:raise ValueError('Diagnostic job is not active')
            (job/'STOP').touch();result={'stop_requested':True,'job_dir':str(job)}
        else:
            report=Path(result['output_dir'])/'report.json'
            if report.is_file():
                saved=json.loads(report.read_text());result['completed_cases']=saved['completed_cases']
    elif command=='summary':
        print((Path(args['output'])/'summary.md').read_text());return 0
    elif command=='_worker':
        job=Path(args['job']).resolve();state=json.loads((job/'status.json').read_text())
        os.chdir(state['cwd'])
        command=[sys.executable,'-u','-m',__package__+'.cfg_diagnostics','_run',str(job/'plan.json')]
        if state['resume']:command.append('--resume')
        commands=[('diagnostics',command)]
        plan=json.loads((job/'plan.json').read_text())
        if plan['schema']=='native-sentence-diagnostics-v1':
            commands.append(('asr',[sys.executable,'-u','-m',__package__+'.sentence_diagnostics',
                                    '_score',str(job/'plan.json')]))
        try:return run_commands(job,state,commands,args['lock_fd'])
        finally:os.close(args['lock_fd'])
    else:
        plan=json.loads(Path(args['plan']).read_text())
        execute(plan,resume=args['resume']);return 0
    print(json.dumps(result,ensure_ascii=False,indent=2));return 0


if __name__=='__main__':
    raise SystemExit(main())

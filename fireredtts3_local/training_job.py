"""Detached Linux training jobs: python -m fireredtts3_local.training_job --help.

Uses the current Python/working directory, a new session and redirected stdio.
No SSH terminal, tmux, systemd, root privileges or new dependencies are needed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def now():
    return datetime.now(timezone.utc).isoformat()


def read_status(job):
    import fcntl
    state = json.loads((job / 'status.json').read_text(encoding='utf-8'))
    with open(state['lock_path'], 'a+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            active = False
        except BlockingIOError:
            lock.seek(0)
            active = lock.read().strip() == str(job.resolve())
    state['active'] = active
    if not active and state['status'] in {'starting', 'running'}:
        state['status'] = 'interrupted'
    return state


def launch(recipe_path, job_dir, precompute=False, stop_after=None):
    import fcntl
    recipe_path, job = Path(recipe_path).resolve(), Path(job_dir).resolve()
    config = json.loads(recipe_path.read_text(encoding='utf-8'))
    output = Path(config['output_dir']).resolve()
    # All jobs writing the same output share one lock, even with different job
    # names. Direct foreground commands are not covered by this launcher lock.
    locks = Path('.runtime/training-job-locks').resolve()
    locks.mkdir(parents=True, exist_ok=True)
    lock_path = locks / (hashlib.sha256(str(output).encode()).hexdigest() + '.lock')
    with lock_path.open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError(f'A background job already owns {output}') from exc
        job.mkdir(parents=True, exist_ok=False)
        lock.seek(0)
        lock.truncate()
        lock.write(str(job))
        lock.flush()
        # Do not depend on a recipe being edited later in another terminal.
        write_json(job / 'recipe.json', config)
        state = {'status': 'starting', 'created_at': now(), 'cwd': str(Path.cwd()),
                 'python': sys.executable, 'source_recipe': str(recipe_path),
                 'output_dir': str(output), 'lock_path': str(lock_path),
                 'log': str(job / 'job.log'), 'precompute': precompute, 'stop_after': stop_after}
        write_json(job / 'status.json', state)
        command = [sys.executable, '-u', '-m', 'fireredtts3_local.training_job',
                   '_worker', str(job), str(lock.fileno())]
        try:
            with (job / 'job.log').open('ab', buffering=0) as log:
                process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                           start_new_session=True, close_fds=True, pass_fds=(lock.fileno(),))
        except BaseException:
            state.update(status='failed', finished_at=now(), error='Could not launch worker')
            write_json(job / 'status.json', state)
            raise
    return {'pid': process.pid, 'job_dir': str(job), 'log': str(job / 'job.log')}


def run_commands(job, state, commands, lock_fd):
    """Worker supervisor; record actual child exit status, including failures."""
    stopping = False
    def request_stop(_signal, _frame):
        nonlocal stopping
        stopping = True
    previous = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    previous[signal.SIGHUP] = signal.signal(signal.SIGHUP, signal.SIG_IGN)
    env = dict(os.environ)
    env.setdefault('OMP_NUM_THREADS', '4')
    env.setdefault('MKL_NUM_THREADS', '4')
    state.update(pid=os.getpid(), started_at=now())
    try:
        for phase, command in commands:
            if stopping or (job / 'STOP').exists():
                state.update(status='stopped', finished_at=now(), exit_code=None)
                write_json(job / 'status.json', state)
                return 0
            state.update(status='running', phase=phase, command=command)
            write_json(job / 'status.json', state)
            print(f'[job] Starting {phase}', flush=True)
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL, env=env,
                                       pass_fds=(lock_fd,) if lock_fd is not None else ())
            state['child_pid'] = process.pid
            write_json(job / 'status.json', state)
            sent_stop = False
            while True:
                if (stopping or (job / 'STOP').exists()) and not sent_stop:
                    sent_stop = True
                    print('[job] Stop requested; asking the child to finish and save.', flush=True)
                    process.terminate()
                try:
                    code = process.wait(timeout=.5)
                    break
                except subprocess.TimeoutExpired:
                    continue
            state.update(exit_code=code)
            if sent_stop or code != 0:
                state.update(status='stopped' if sent_stop else 'failed', finished_at=now())
                write_json(job / 'status.json', state)
                print(f'[job] {state["status"]}: {phase}, exit_code={code}', flush=True)
                return code if code > 0 else (1 if code < 0 and not sent_stop else 0)
        state.update(status='completed', finished_at=now())
        write_json(job / 'status.json', state)
        print('[job] Completed successfully.', flush=True)
        return 0
    except BaseException as exc:
        state.update(status='failed', finished_at=now(), error=str(exc))
        write_json(job / 'status.json', state)
        raise
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    start = commands.add_parser('start', help='Detach precompute (optional) and training from SSH')
    start.add_argument('recipe')
    start.add_argument('--job-dir', required=True, help='New directory for recipe snapshot, log and status')
    start.add_argument('--precompute', action='store_true', help='Precompute missing features before training')
    start.add_argument('--stop-after', type=int, help='Pilot: stop after N training updates and save')
    for name in ('status', 'stop'):
        sub = commands.add_parser(name)
        sub.add_argument('job_dir')
    worker = commands.add_parser('_worker', help=argparse.SUPPRESS)
    worker.add_argument('job_dir')
    worker.add_argument('lock_fd', type=int)
    args = parser.parse_args()
    if os.name != 'posix':
        parser.error('Detached jobs currently require Linux/POSIX')
    if args.command == 'start':
        if args.stop_after is not None and args.stop_after < 1:
            parser.error('--stop-after must be positive')
        result = launch(args.recipe, args.job_dir, args.precompute, args.stop_after)
    elif args.command in {'status', 'stop'}:
        job = Path(args.job_dir).resolve()
        result = read_status(job)
        if args.command == 'stop':
            if not result['active']:
                parser.error('Job is not active')
            (job / 'STOP').touch()
            result = {'stop_requested': True, 'job_dir': str(job)}
    else:
        job = Path(args.job_dir).resolve()
        state = json.loads((job / 'status.json').read_text(encoding='utf-8'))
        os.chdir(state['cwd'])
        prefix = [sys.executable, '-u', '-m', 'fireredtts3_local.adapt']
        recipe = str(job / 'recipe.json')
        stages = [('precompute', prefix + ['precompute', recipe])] if state['precompute'] else []
        train = prefix + ['train', recipe]
        if state['stop_after'] is not None:
            train += ['--stop-after', str(state['stop_after'])]
        stages.append(('train', train))
        try:
            return run_commands(job, state, stages, args.lock_fd)
        finally:
            os.close(args.lock_fd)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

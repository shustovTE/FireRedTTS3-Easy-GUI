"""Trace one latent perturbation through native FireRed generation using a saved listening pack."""
import argparse
import csv
import json
import math
from contextlib import contextmanager
from dataclasses import dataclass, field
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from filelock import FileLock
import numpy as np
import soundfile as sf
import torch

from . import redae_decoder_training
from .model_manager import ModelManager
from .paths import MODEL_DIR
from .tail_contrast import contrast_trace, contrast_windows
from .training_data import file_digest, model_identity
from .training_job import write_json
from .training_monitor import SETTINGS, generate_case
from .training_objective import BACKBONE_ATTENTION
from .vox_attention_ablation import checkpoint_files, route
from .vox_recipe_transfer import BASE_SHA


ROUTES = ('dit_history', 'backbone_history', 'both_histories')
DEFAULT_CASES = [f'cities_seed{s}' for s in range(1234, 1240)] + ['timofey_seed1234', 'timofey_seed1235']


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


@dataclass
class Trace:
    patches: list = field(default_factory=list)
    rng: list = field(default_factory=list)
    dit_reads: list = field(default_factory=list)
    backbone_reads: list = field(default_factory=list)


def make_delta(anchor, fraction, seed):
    """One reproducible latent direction, independent of the generator's noise stream."""
    local = torch.Generator(device='cpu').manual_seed(seed)
    direction = torch.randn(anchor.shape, generator=local, dtype=torch.float32)
    return direction * (fraction * anchor.square().mean().sqrt() / direction.square().mean().sqrt())


@contextmanager
def observe_feedback(core, anchor, mode='control', *, control=None, delta=None):
    if mode not in ('control', 'sham', *ROUTES):
        raise ValueError(f'Unknown feedback route: {mode}')
    trace = Trace()
    native = core._flow_one_step
    pending = False

    def flow(*, hist_latents, **kwargs):
        nonlocal pending
        i = len(trace.patches)
        device = hist_latents.device
        # The first op in native _flow_one_step samples x0 on this device.
        state = torch.cuda.get_rng_state(device) if device.type == 'cuda' else torch.get_rng_state()
        if control is not None and i < len(control.rng) and not torch.equal(state, control.rng[i]):
            raise RuntimeError(f'DiT noise stream differs from control at patch {i}')
        trace.rng.append(state)
        lag = i - anchor
        if mode in ('dit_history', 'both_histories') and 1 <= lag <= core.history_patches:
            # Same single historical patch as it slides through the native history buffer.
            start = hist_latents.shape[1] - lag * core.patch_size
            hist_latents = hist_latents.clone()
            hist_latents[:, start:start+core.patch_size] += delta.to(hist_latents)
            trace.dit_reads.append(i)
        value = native(hist_latents=hist_latents, **kwargs)
        emitted = value.detach().float().cpu().clone()
        if not torch.isfinite(emitted).all():
            raise RuntimeError(f'Nonfinite latent at patch {i}')
        if control is not None and (i <= anchor or mode == 'sham'):
            if i >= len(control.patches) or not torch.equal(emitted, control.patches[i]):
                raise RuntimeError(f'Unperturbed native replay diverged at patch {i}')
        trace.patches.append(emitted)
        pending = i == anchor and mode in ('backbone_history', 'both_histories')
        return value

    def patch_input(_module, args):
        nonlocal pending
        if pending:
            pending = False
            trace.backbone_reads.append(len(trace.patches)-1)
            return (args[0] + delta.to(args[0]),)

    hook = core.patch_encoder.register_forward_pre_hook(patch_input)
    try:
        with patch.object(core, '_flow_one_step', flow):
            yield trace
            if pending:
                raise RuntimeError('Native PatchEncoder did not receive the anchor')
    finally:
        hook.remove()


def paired_latents(control, trace, anchor, delta, dt):
    if len(trace.patches) <= anchor:
        raise RuntimeError('Branch stopped before the shared anchor')
    size = float(delta.square().mean().sqrt())
    if not math.isfinite(size) or size <= 0:
        raise ValueError('The selected latent perturbation has zero or nonfinite RMS')
    rows = []
    for i in range(anchor + 1, min(len(control.patches), len(trace.patches))):
        difference = trace.patches[i] - control.patches[i]
        rms = float(difference.square().mean().sqrt())
        rows.append(dict(hop=i-anchor, patch=i, start_s=i*dt, delta_rms=rms, gain=rms/size))
    return rows


def paired_windows(control, wave, rate):
    length = min(len(control), len(wave))
    a, b = control[:length], wave[:length]
    times, reference = contrast_trace(a, rate)
    _, variant = contrast_trace(b, rate)
    rows = []
    for start in range(0, length, rate):
        end = min(start+rate, length)
        keep = (times >= start/rate) & (times < end/rate) & np.isfinite(reference) & np.isfinite(variant)
        def rms_dbfs(x):
            return 10 * math.log10(max(float(np.mean(np.square(x[start:end], dtype=np.float64))), 1e-24))
        control_level, branch_level = rms_dbfs(a), rms_dbfs(b)
        rows.append(dict(start_s=start/rate, end_s=end/rate, control_rms_dbfs=control_level,
                         branch_rms_dbfs=branch_level, rms_delta_db=branch_level-control_level,
                         control_resonance_db=float(np.median(reference[keep])) if keep.any() else None,
                         branch_resonance_db=float(np.median(variant[keep])) if keep.any() else None,
                         paired_active_frames=int(keep.sum()), resonance_delta_db=(
                             float(np.median(variant[keep]-reference[keep])) if keep.any() else None)))
    return rows


def write_csv(path, columns, rows):
    temporary = path.with_suffix('.csv.tmp')
    with temporary.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def replay_case(tts, case, settings, original, folder, anchor, fraction):
    core = tts.tts_core.get_base_model()
    dt = core.patch_size * tts.redae.downsample_rate / tts.redae.sample_rate
    def render(mode, **kwargs):
        with observe_feedback(core, anchor, mode, **kwargs) as trace:
            samples, rate, run = generate_case(tts, case, settings)
        if len(trace.patches) != run['generated_patches'] or rate != tts.redae.sample_rate:
            raise RuntimeError('Expected one native flow call per emitted patch and native output rate')
        return samples, trace, run

    audio, control, run = render('control')
    rate = run['sample_rate']
    archived, archive_rate = sf.read(original, dtype='float32')
    exact = rate == archive_rate and np.array_equal(audio, archived)
    sf.write(folder/'control.wav', audio, rate, subtype='FLOAT')
    summary = dict(case=case['id'], seed=case['seed'], anchor_patch=anchor,
                   anchor_start_s=anchor*dt, first_affected_s=(anchor+1)*dt,
                   archive_pcm_exact=exact, archive_seconds=len(archived)/archive_rate,
                   control_seconds=len(audio)/rate, control_patches=len(control.patches),
                   control_finish=run['finish_reason'], control_spectrum=contrast_windows(audio, rate),
                   branches={})
    if not exact:
        print(f"[feedback] {case['id']}: archived PCM differs; comparisons use the fresh control.wav", flush=True)
    if len(control.patches) <= anchor+1:
        return dict(summary, status='insufficient_continuation')

    # Establish the numerical noise floor across the entire utterance, including its stop decision.
    sham, sham_trace, sham_run = render('sham', control=control)
    if (not np.array_equal(sham, audio) or len(sham_trace.patches) != len(control.patches)
            or sham_run['stop_scores'] != run['stop_scores']):
        raise RuntimeError('Native repeat at zero perturbation differs; paired results are not interpretable')
    delta = make_delta(control.patches[anchor], fraction, case['seed'])
    summary.update(sham_exact=True, perturbation_rms=float(delta.square().mean().sqrt()))
    patch_rows, audio_rows = [], []
    for mode in ROUTES:
        wave, trace, branch = render(mode, control=control, delta=delta)
        pairs = paired_latents(control, trace, anchor, delta, dt)
        windows = paired_windows(audio, wave, rate)
        sf.write(folder/f'{mode}.wav', wave, rate, subtype='FLOAT')
        patch_rows.extend(dict(route=mode, **row) for row in pairs)
        audio_rows.extend(dict(route=mode, **row) for row in windows)
        gain_after_history = [r['gain'] for r in pairs if core.history_patches < r['hop'] <= core.history_patches+10]
        tail_start = max(summary['first_affected_s'], windows[-1]['end_s']*2/3)
        tail = [r['resonance_delta_db'] for r in windows
                if r['start_s'] >= tail_start and r['resonance_delta_db'] is not None]
        summary['branches'][mode] = dict(seconds=len(wave)/rate, finish=branch['finish_reason'],
            generated_patches=len(trace.patches), compared_hops=len(pairs), dit_input_reads=trace.dit_reads,
            backbone_input_reads=trace.backbone_reads, spectrum=contrast_windows(wave, rate),
            median_gain_after_direct_history=float(np.median(gain_after_history)) if gain_after_history else None,
            paired_tail_resonance_delta_db=float(np.median(tail)) if tail else None,
            max_gain=max((r['gain'] for r in pairs), default=None))
        print(f"[feedback] {case['id']} {mode}: {len(pairs)} paired hops, {len(wave)/rate:.2f}s", flush=True)
    write_csv(folder/'patches.csv', ('route','hop','patch','start_s','delta_rms','gain'), patch_rows)
    write_csv(folder/'windows.csv', ('route','start_s','end_s','control_rms_dbfs','branch_rms_dbfs',
                                     'rms_delta_db','control_resonance_db','branch_resonance_db',
                                     'paired_active_frames','resonance_delta_db'), audio_rows)
    return dict(summary, status='tested')


def resolve_checkpoint(path):
    path = path.resolve()
    return path if path.parent.name == 'lora' else path/'lora'/'step-0500'


def load_inputs(checkpoint, ids):
    pack = checkpoint.parent.parent/'audio'/checkpoint.name/'backbone-complete'
    report = read_json(pack/'report.json')
    plan = report['plan']
    profile = plan['adapter_profile']
    if (not report['audio_complete'] or profile not in ('vox_attention', BACKBONE_ATTENTION)
            or plan['variants'] != ['backbone'] or plan.get('redae_decoder')
            or redae_decoder_training.has_decoder(checkpoint) or plan['settings'] != SETTINGS
            or plan['attention'] != 'sdpa' or plan['base_sha256'] != BASE_SHA):
        raise ValueError('Expected an attention checkpoint and its native backbone-only listening pack')
    expected = plan['checkpoints'][checkpoint.name]['hashes']
    if checkpoint_files(checkpoint)[checkpoint.name]['hashes'] != expected:
        raise ValueError('Checkpoint files differ from the original listening run')
    record = read_json(checkpoint/'checkpoint.json')
    if (profile != record['recipe'].get('adapter_profile', 'vox_attention')
            or checkpoint.name != f"step-{record['step']:04d}"):
        raise ValueError('Listening profile or step differs from the training checkpoint')
    available = {case['id']:case for case in plan['cases']}
    selected = list(available) if ids == ['all'] else ids
    if len(set(selected)) != len(selected) or set(selected)-available.keys():
        raise ValueError(f'Choose unique cases from {list(available)}')
    cases = []
    for name in selected:
        case = available[name]
        relative = Path(case['filename'])
        if relative.is_absolute() or '..' in relative.parts or relative.suffix != '.wav':
            raise ValueError(f'Invalid listening filename for {name}')
        saved = report['results'][case['filename']]
        if (saved['case'] != name or saved['active_layers']['dit'] != 0 or saved['active_layers']['backbone'] <= 0
                or file_digest(case['reference_audio']) != case['reference_sha256']
                or file_digest(pack/case['filename']) != saved['wav_sha256']):
            raise ValueError(f'Archived recording or reference differs for {name}')
        cases.append(dict(case, archive_sha256=saved['wav_sha256']))
    return pack, dict(checkpoint=str(checkpoint), weights=expected, base_sha256=BASE_SHA, adapter_profile=profile,
                      settings=plan['settings'], cases=cases,
                      runtime={p:version(p) for p in ('torch','torchaudio','peft','transformers')})


def render_pack(output, pack, plan):
    output.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output/'.probe.lock'), timeout=0):
        plan_path = output/'plan.json'
        if plan_path.exists():
            if read_json(plan_path) != plan:
                raise ValueError(f'Inputs differ from {plan_path}; use a new output directory')
        elif any(p.name != '.probe.lock' for p in output.iterdir()):
            raise ValueError(f'Choose an empty output directory: {output}')
        else:
            write_json(plan_path, plan)
        manager = ModelManager()
        tts = None
        results = []
        try:
            for case in plan['cases']:
                folder = output/Path(case['filename']).with_suffix('')
                if folder.is_dir():
                    saved = read_json(folder/'result.json')
                    if any(file_digest(folder/f) != h for f,h in saved['files'].items()):
                        raise ValueError(f'Existing probe files have changed: {folder}')
                else:
                    if tts is None:
                        tts = manager.load_base(attention_backend='sdpa', adapter_path=plan['checkpoint'])
                        counts = route(tts.tts_core, 'backbone', backbone_only=plan['adapter_profile'] == BACKBONE_ATTENTION)
                        print(f"[feedback] trained={plan['adapter_profile']} active={counts}", flush=True)
                    dt = tts.tts_core.patch_size*tts.redae.downsample_rate/tts.redae.sample_rate
                    anchor = math.floor(plan['at_seconds']/dt + .5)
                    print(f"[feedback] {case['id']}: native repeat and feedback tests at {anchor*dt:.2f}s", flush=True)
                    folder.parent.mkdir(parents=True, exist_ok=True)
                    with TemporaryDirectory(dir=folder.parent, prefix='.feedback-') as staging:
                        tmp = Path(staging)
                        saved = replay_case(tts, case, plan['settings'], pack/case['filename'],
                                            tmp, anchor, plan['fraction'])
                        saved['files'] = {p.name:file_digest(p) for p in tmp.iterdir()}
                        write_json(tmp/'result.json', saved)
                        tmp.rename(folder)
                results.append(saved)
                print(f"[feedback] {case['id']}: {saved['status']} -> {folder}", flush=True)
            write_json(output/'report.json', dict(complete=True, plan=plan, results=results))
        finally:
            manager.unload()
    print(output/'report.json', flush=True)
    return results


def validate_pair(plans):
    if {p['adapter_profile'] for p in plans} != {'vox_attention', BACKBONE_ATTENTION}:
        raise ValueError('Paired comparison needs one joint-trained and one backbone-only checkpoint')
    def signature(p):
        keys = ('id', 'seed', 'text', 'language', 'reference_text', 'reference_sha256', 'filename')
        return (Path(p['checkpoint']).name, p['settings'], [[c[k] for k in keys] for c in p['cases']])
    if signature(plans[0]) != signature(plans[1]):
        raise ValueError('Step, text, reference, seed or native settings differ between the listening packs')


def compare_runs(output, cases, results):
    columns = ('case','route','status','common_hops','joint_archive_exact','backbone_only_archive_exact',
               'joint_control_s','backbone_only_control_s','joint_branch_s','backbone_only_branch_s',
               'matched_tail_windows','joint_tail_delta_db','backbone_only_tail_delta_db',
               'joint_gain_hops3_12','backbone_only_gain_hops3_12')
    rows = []
    for index, case in enumerate(cases):
        pair = [results[k][index] for k in ('joint','backbone-only')]
        head = dict(case=case['id'], joint_archive_exact=pair[0]['archive_pcm_exact'],
                    backbone_only_archive_exact=pair[1]['archive_pcm_exact'], joint_control_s=pair[0]['control_seconds'],
                    backbone_only_control_s=pair[1]['control_seconds'])
        if any(r['status'] != 'tested' for r in pair):
            rows.append(dict(head, status='insufficient_continuation'))
            continue
        tables = []
        for label in ('joint','backbone-only'):
            folder = output/label/Path(case['filename']).with_suffix('')
            with (folder/'windows.csv').open(newline='') as f:
                tables.append(list(csv.DictReader(f)))
        for mode in ROUTES:
            branches = [r['branches'][mode] for r in pair]
            common = min(b['compared_hops'] for b in branches)
            common_end = min(head['joint_control_s'], head['backbone_only_control_s'], *(b['seconds'] for b in branches))
            tail_start = max(common_end*2/3, *(r['first_affected_s'] for r in pair))
            # Match actual window boundaries; a partial terminal window can differ across branches.
            bands = [{(r['start_s'],r['end_s']):float(r['resonance_delta_db']) for r in table
                      if r['route'] == mode and r['resonance_delta_db'] and
                      float(r['start_s']) >= tail_start and float(r['end_s']) <= common_end} for table in tables]
            matched = bands[0].keys() & bands[1].keys()
            tails = [float(np.median([b[k] for k in matched])) if matched else None for b in bands]
            gains = [b['median_gain_after_direct_history'] if common >= 12 else None for b in branches]
            rows.append(dict(head, route=mode, status='tested', common_hops=common,
                             joint_branch_s=branches[0]['seconds'], backbone_only_branch_s=branches[1]['seconds'],
                             matched_tail_windows=len(matched), joint_tail_delta_db=tails[0],
                             backbone_only_tail_delta_db=tails[1], joint_gain_hops3_12=gains[0],
                             backbone_only_gain_hops3_12=gains[1]))
    write_csv(output/'comparison.csv', columns, rows)


def run(checkpoint, output, cases, at_seconds, fraction, compare_run=None):
    if not math.isfinite(at_seconds) or at_seconds < 0 or not 0 < fraction <= 1:
        raise ValueError('Require a finite nonnegative anchor time and 0 < fraction <= 1')
    print('[feedback] Verifying Base, checkpoints and archived WAVs', flush=True)
    if model_identity(MODEL_DIR) != BASE_SHA:
        raise ValueError('Installed Base differs from listening checkpoint')
    output = output.resolve()
    inputs = []
    for target in ([checkpoint, compare_run] if compare_run else [checkpoint]):
        pack, plan = load_inputs(resolve_checkpoint(target), cases)
        plan.update(schema='native-feedback-routes-v1', at_seconds=at_seconds, fraction=fraction)
        inputs.append((pack, plan))
    if compare_run is None:
        return render_pack(output, *inputs[0])
    plans = [plan for _, plan in inputs]
    validate_pair(plans)
    output.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output/'.pair.lock'), timeout=0):
        pair_plan = {('joint' if p['adapter_profile']=='vox_attention' else 'backbone-only'):p for p in plans}
        path = output/'pair.json'
        if path.exists():
            if read_json(path) != pair_plan:
                raise ValueError(f'Paired inputs changed; use a new output directory: {output}')
        elif any(p.name != '.pair.lock' for p in output.iterdir()):
            raise ValueError(f'Choose an empty paired output directory: {output}')
        else:
            write_json(path, pair_plan)
        results = {}
        for pack, plan in inputs:
            label = 'joint' if plan['adapter_profile']=='vox_attention' else 'backbone-only'
            results[label] = render_pack(output/label, pack, plan)
        compare_runs(output, plans[0]['cases'], results)
        print(f"[feedback] Both checkpoints completed: {output/'comparison.csv'}", flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path, help='RUN or RUN/lora/step-NNNN with a backbone listening pack')
    parser.add_argument('--compare-run', type=Path, help='Second RUN or step directory for a paired comparison')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cases', nargs='+', default=DEFAULT_CASES, help='Saved case IDs, or all')
    parser.add_argument('--at-seconds', type=float, default=12., help='Start of the single perturbed patch')
    parser.add_argument('--fraction', type=float, default=.02, help='Injected latent RMS / original patch RMS')
    run(**vars(parser.parse_args()))

"""Accept provisional LoRA updates using paired, native, whole-utterance RMS measurements."""
import copy
import json
import math
import time

import torch

from .history_probe import levels
from .training_batch import set_training
from .training_job import write_json
from .training_monitor import SETTINGS, generate_case, load_benchmark


DEFAULT_CASE = 'cities_seed1237'


def configuration(percent, ids, reports):
    """Bind each control to its exact reference, seed and measured original Base drift."""
    from .vox_recipe_compare import load_saved_base
    if not math.isfinite(percent) or percent < 0:
        raise ValueError('The rise tolerance must be a finite nonnegative amplitude percentage')
    if not ids or len(set(ids)) != len(ids):
        raise ValueError('Supply distinct case IDs for the per-update rise check')
    cases = {}
    for report in reports:
        _, baseline = load_saved_base(report)
        rows, _ = load_benchmark(report.parent / 'benchmark.jsonl')
        for row in rows:
            key = row['id']
            if key not in ids:
                continue
            case = dict(row, base_rise_db=baseline[key]['end_minus_start_rms_db'])
            if key in cases and cases[key] != case:
                raise ValueError(f'Conflicting saved Base controls for {key}')
            cases[key] = case
    missing = set(ids) - cases.keys()
    if missing:
        raise ValueError(f'Missing requested Base controls: {sorted(missing)}')
    return dict(schema='paired-rms-rise-v1', percent=percent, margin_db=20 * math.log1p(percent/100) / math.log(10),
                settings=dict(SETTINGS), cases=[cases[k] for k in ids])


@torch.inference_mode()
def evaluate(tts, config):
    core = tts.tts_core
    was_training = core.training
    results = []
    try:
        set_training(core, False)
        for case in config['cases']:
            # Same unsplit generator, isolated seed, raw FLOAT waveform and nonoverlapping
            # head/tail windows as the saved Base report. The reference is fixed for each ID.
            samples, rate, generation = generate_case(tts, case, settings=config['settings'])
            current = levels(samples, rate)['end_minus_start_rms_db']
            delta = current - case['base_rise_db']
            if not math.isfinite(delta):
                raise FloatingPointError(f'Invalid paired rise for {case["id"]}')
            results.append(dict(id=case['id'], base_rise_db=case['base_rise_db'], rise_db=current,
                                excess_db=delta, passed=delta <= config['margin_db'],
                                seconds=len(samples)/rate, finish_reason=generation['finish_reason']))
    finally:
        set_training(core, was_training)
    return results


def try_update(optimizer, params, tts, config):
    # Includes AdamW moments, step counters, weight decay effects and lazy initial state.
    previous = [p.detach().clone() for p in params]
    previous_optimizer = copy.deepcopy(optimizer.state_dict())
    accepted = False
    started = time.monotonic()
    try:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        results = evaluate(tts, config)
        accepted = all(r['passed'] for r in results)
        return dict(accepted=accepted, cases=results, check_seconds=time.monotonic()-started)
    finally:
        if not accepted:
            with torch.no_grad():
                for p, saved in zip(params, previous, strict=True):
                    p.copy_(saved)
            optimizer.load_state_dict(previous_optimizer)


class RiseLog:
    """An atomic per-attempt audit. Discard tentative rows after the restored checkpoint."""
    def __init__(self, output, start):
        self.path = output / 'rise_decisions.json'
        self.rows = []
        if start and self.path.exists():
            self.rows = [r for r in json.loads(self.path.read_text()) if r['attempt'] <= start]
        write_json(self.path, self.rows)

    def record(self, attempt, accepted_total, decision):
        row = dict(attempt=attempt, accepted_total=accepted_total, rejected_total=attempt-accepted_total, **decision)
        self.rows.append(row)
        write_json(self.path, self.rows)
        measurements = ', '.join(f"{r['id']}={r['excess_db']:+.3f}dB" for r in row['cases'])
        action = 'accepted' if row['accepted'] else 'rejected'
        print(f"[rise] attempt={attempt} {action} accepted_total={accepted_total} "
              f"excess_vs_base {measurements} check={row['check_seconds']:.2f}s", flush=True)

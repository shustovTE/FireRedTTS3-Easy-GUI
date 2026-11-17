"""Construct a reproducible full epoch from unique, VAD-checked training pairs."""
from collections import defaultdict
import hashlib
import json
import math

from .yo_dataset import has_sentence_ending

SECONDS = range(3, 31)
MANIFEST = "balanced-train.jsonl"


def plan(rows):
    """Retain each target once, then assign additional draws in whole recordings."""
    groups = defaultdict(lambda: {True: [], False: []})
    for index, row in enumerate(rows):
        duration = row["duration"]
        if not 3 <= duration <= 30 or not has_sentence_ending(row["text"].replace("“", "”")):
            raise ValueError("Training pairs require complete punctuation and 3..30 seconds after VAD")
        groups[math.floor(duration + .5)]["?" in row["text"]].append(index)
    if not groups:
        raise ValueError("No eligible training pairs remain")

    counts = {s: sum(map(len, g.values())) for s, g in groups.items()}
    quota = max(counts.values())
    budgets = dict.fromkeys(counts, quota)
    transfers = {}
    for s in SECONDS:
        if s in counts:
            continue
        left = max((x for x in counts if x < s), default=None)
        right = min((x for x in counts if x > s), default=None)
        if left is None or right is None:
            shares = {right if left is None else left: quota}
        else:
            # Alternate the odd extra draw across successive gaps with the same neighbours.
            extra_left = (s - left) % 2
            to_left = (quota + extra_left) // 2
            shares = {left: to_left, right: quota - to_left}
        transfers[str(s)] = {str(k): v for k, v in shares.items()}
        for neighbour, amount in shares.items():
            budgets[neighbour] += amount

    # First distribute repeats proportionally to the unique question/statement mix.
    # Promote only repeat slots if more questions are needed. A unique recording
    # is never evicted to meet the question floor.
    total = sum(budgets.values())
    required = (3 * total + 9) // 10
    questions, capacity = {}, {}
    for s in sorted(groups):
        q = len(groups[s][True])
        spare = budgets[s] - counts[s]
        questions[s] = q + (spare * q + counts[s] // 2) // counts[s]
        capacity[s] = q + (spare if q else 0) - questions[s]
    deficit = max(0, required - sum(questions.values()))
    available = sum(capacity.values())
    if deficit > available:
        raise ValueError(f"Question floor infeasible: need {required}/{total}, maximum "
                         f"{sum(questions.values()) + available}/{total} while keeping every recording "
                         f"and the requested length budgets. Unique questions per second: "
                         f"{dict(sorted((s, len(g[True])) for s, g in groups.items()))}")
    if deficit:
        allocated = {s: divmod(deficit * capacity[s], available) for s in groups}
        remainder = deficit - sum(base for base, _ in allocated.values())
        bonus = set(sorted(groups, key=lambda s: (-allocated[s][1], s))[:remainder])
        for s, (base, _) in allocated.items():
            questions[s] += base + int(s in bonus)

    repeats = [0] * len(rows)
    buckets = {}
    for s in sorted(groups):
        for is_question, indices in groups[s].items():
            draws = questions[s] if is_question else budgets[s] - questions[s]
            if not indices:
                continue
            # Use the content-derived WAV filename so relocating a checkout does
            # not influence which recordings receive the remainder.
            order = sorted(indices, key=lambda i: hashlib.sha256(rows[i]["audio"].rsplit("/", 1)[-1].encode()).digest())
            base, extra = divmod(draws, len(order))
            for position, index in enumerate(order):
                repeats[index] = base + int(position < extra)
        buckets[str(s)] = {"unique": counts[s], "unique_questions": len(groups[s][True]), "draws": budgets[s],
                           "question_draws": questions[s], "max_draws_per_clip": max(
                               repeats[i] for indices in groups[s].values() for i in indices)}
    if min(repeats) < 1 or sum(repeats) != total or sum(questions.values()) < required:
        raise RuntimeError("Inconsistent length/question allocation")
    return repeats, {"unique_targets": len(rows), "target_draws": total, "base_draws_per_second": quota,
                     "question_draws": sum(questions.values()), "minimum_question_draws": required,
                     "question_fraction": sum(questions.values()) / total,
                     "empty_bin_transfers": transfers, "buckets": buckets}


def manifest_digest(rows, repeats, stream=None):
    """Emit repeat rows as ordinary upstream JSONL, without duplicating any WAV."""
    digest = hashlib.sha256()
    for row, n in zip(rows, repeats, strict=True):
        line = (json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode()
        # Bound allocations even if one rare recording must occupy thousands of slots.
        while n:
            block_rows = min(n, 256)
            block = line * block_rows
            digest.update(block)
            if stream is not None:
                stream.write(block)
            n -= block_rows
    return digest.hexdigest()

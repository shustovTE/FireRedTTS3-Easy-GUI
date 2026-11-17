"""Construct a reproducible full epoch from unique, VAD-checked training pairs."""
from collections import Counter, defaultdict
import hashlib
import json
import math

from .yo_dataset import has_sentence_ending

SECONDS = range(3, 31)
MANIFEST = "balanced-train.jsonl"
FULL_DRAWS = 1000 * 2 * 8


def _groups(rows):
    groups = defaultdict(lambda: {True: [], False: []})
    for index, row in enumerate(rows):
        duration = row["duration"]
        if not 3 <= duration <= 30 or not has_sentence_ending(row["text"].replace("“", "”")):
            raise ValueError("Training pairs require complete punctuation and 3..30 seconds after VAD")
        groups[math.floor(duration + .5)]["?" in row["text"]].append(index)
    if not groups:
        raise ValueError("No eligible training pairs remain")

    return groups


def plan(rows):
    """Retain each target once, then assign additional draws in whole recordings."""
    groups = _groups(rows)
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


def even_repeats(rows, corpora, *, draws=FULL_DRAWS):
    """Allocate the fixed single-GPU budget with the smallest clip exposure disparity.

    At the natural question share, all clip counts differ by at most one.
    If the floor binds, use as few extra question slots as necessary and keep
    counts within each class at floor/ceil. Duration only decides remainders.
    """
    groups = _groups(rows)
    if len(corpora) != len(rows) or len(rows) > draws:
        raise ValueError("Fixed draw budget must cover each identified training target")
    n_questions = sum(len(group[True]) for group in groups.values())
    required = (3 * draws + 9) // 10
    if not n_questions or draws - (len(rows) - n_questions) < required:
        raise ValueError("Cannot reach 30% questions while presenting every target at least once")
    q_draws = max(required, (draws * n_questions + len(rows) // 2) // len(rows))
    counts = Counter()
    repeats = [0] * len(rows)
    remaining = {}
    for question, budget in ((True, q_draws), (False, draws - q_draws)):
        n = n_questions if question else len(rows) - n_questions
        base, remaining[question] = divmod(budget, n) if n else (0, 0)
        for sec, group in groups.items():
            group[question].sort(key=lambda i: hashlib.sha256(rows[i]["audio"].rsplit("/", 1)[-1].encode()).digest())
            for i in group[question]:
                repeats[i] = base
            counts[sec] += base * len(group[question])

    # Each second has two half-shares. Transfer a missing second's half-shares
    # to its closest occupied neighbours, both to the sole neighbour at an edge.
    weights = Counter()
    for sec in SECONDS:
        left = max((s for s in groups if s <= sec), default=None)
        right = min((s for s in groups if s >= sec), default=None)
        weights[left or right] += 1
        weights[right or left] += 1
    units = 2 * len(SECONDS)
    while any(remaining.values()):
        eligible = ((sec, q) for sec, group in groups.items() for q in (True, False)
                    if remaining[q] and group[q])
        # Prefer the duration farthest below its share, within the clip budgets.
        sec, question = max(eligible, key=lambda k: (draws * weights[k[0]] - units * counts[k[0]],
                                                     remaining[k[1]], -k[0], k[1]))
        i = groups[sec][question].pop()
        repeats[i] += 1
        counts[sec] += 1
        remaining[question] -= 1

    histogram = Counter(repeats)
    class_counts = defaultdict(Counter)
    corpus_counts = Counter()
    buckets = defaultdict(list)
    for row, corpus, times in zip(rows, corpora, repeats, strict=True):
        buckets[math.floor(row["duration"] + .5)].append(times)
        corpus_counts[corpus] += times
        class_counts["questions" if "?" in row["text"] else "other"][str(times)] += 1
    if min(repeats) < 1 or sum(repeats) != draws or sum(
            times for row, times in zip(rows, repeats) if "?" in row["text"]) != q_draws:
        raise RuntimeError("Inconsistent per-record exposure or question allocation")
    return repeats, {"unique_targets": len(rows), "target_draws": draws, "unique_questions": n_questions,
                     "question_draws": q_draws, "minimum_question_draws": required,
                     "question_fraction": q_draws / draws, "corpus_draws": dict(corpus_counts),
                     "draw_histogram_by_type": {k: dict(v) for k, v in class_counts.items()},
                     "draw_histogram": {str(k): v for k, v in sorted(histogram.items())},
                     "min_draws_per_clip": min(repeats), "max_draws_per_clip": max(repeats),
                     "empty_seconds": [s for s in SECONDS if s not in groups],
                     "buckets": {str(s): {"unique": len(buckets[s]), "draws": counts[s],
                                          "desired_draws": draws * weights[s] / units}
                                 for s in SECONDS}}


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

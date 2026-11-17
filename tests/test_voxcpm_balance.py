import hashlib
import io
import json
import unittest
from collections import Counter

from fireredtts3_local.voxcpm_balance import even_repeats, manifest_digest, plan


class LengthAndQuestionBalanceTests(unittest.TestCase):
    def test_boundary_rounding_neighbour_transfers_question_floor_and_exact_manifest(self):
        durations = [3., 3.49, 3., 3.5, 6., 6., 6., 30.]
        rows = [{"audio": f"{i}.wav", "duration": d,
                 "text": "Готовы? Тогда начнём." if i in {0, 4, 7} else "Готово."} for i, d in enumerate(durations)]
        repeats, report = plan(rows)
        self.assertGreaterEqual(min(repeats), 1)
        self.assertEqual(report["buckets"]["3"]["unique"], 3)
        self.assertEqual(report["empty_bin_transfers"]["5"], {"4": 2, "6": 1})
        self.assertEqual(sum(repeats), 28 * 3)
        self.assertGreaterEqual(sum(n for r, n in zip(rows, repeats) if "?" in r["text"]), .3 * sum(repeats))
        out = io.BytesIO()
        self.assertEqual(manifest_digest(rows, repeats, out), hashlib.sha256(out.getvalue()).hexdigest())
        self.assertEqual(len([json.loads(line) for line in out.getvalue().splitlines()]), sum(repeats))

    def test_promotes_questions_using_only_repeats_and_handles_an_empty_edge(self):
        rows = [{"audio": str(i), "duration": 4, "text": "Да."} for i in range(8)]
        rows += [{"audio": "q", "duration": 10, "text": "Да?"}, {"audio": "a", "duration": 10, "text": "Да."},
                 {"audio": "b", "duration": 10, "text": "Да."}, {"audio": "c", "duration": 10, "text": "Да."}]
        repeats, report = plan(rows)
        self.assertEqual(report["empty_bin_transfers"]["3"], {"4": 8})
        self.assertEqual(report["empty_bin_transfers"]["30"], {"10": 8})
        self.assertEqual(repeats[8], report["minimum_question_draws"])
        self.assertEqual(report["question_draws"], repeats[8])
        self.assertEqual(report["buckets"]["4"]["draws"] + report["buckets"]["10"]["draws"], 224)
        with self.assertRaisesRegex(ValueError, "Question floor infeasible"):
            plan(rows[:8])

    def test_rejects_ellipsis_or_unfinished_text_and_outside_range(self):
        for text, duration in [("Обрыв", 8), ("Обрыв…", 8), ("Обрыв?..", 8), ("Да?", 2.99), ("Да?", 30.01)]:
            with self.subTest(text=text, duration=duration), self.assertRaises(ValueError):
                plan([{"audio": "test.wav", "duration": duration, "text": text}])


class ClipExposureTests(unittest.TestCase):
    def test_rare_duration_does_not_override_equal_clip_repeats(self):
        rows = [{"audio": f"{i}.wav", "duration": 4 if i < 3998 else 29.9,
                 "text": "Готовы? Начинаем." if i % 2 else "Хорошо."} for i in range(4001)]
        corpora = ["dialogs"] * 3701 + ["spontaneous"] * 300
        repeats, report = even_repeats(rows, corpora)
        self.assertEqual(sum(repeats), 16000)
        self.assertEqual(set(repeats), {3, 4})
        self.assertEqual(repeats[-3:], [4, 4, 4])
        self.assertEqual(report["corpus_draws"]["spontaneous"], sum(repeats[-300:]))
        self.assertEqual({str(k): v for k, v in Counter(repeats).items()}, report["draw_histogram"])
        self.assertGreaterEqual(report["question_draws"], 4800)
        self.assertIn(29, report["empty_seconds"])
        reverse, _ = even_repeats(rows[::-1], corpora[::-1])
        self.assertEqual(repeats, reverse[::-1])

    def test_question_constraint_changes_only_necessary_exposures(self):
        rows = [{"audio": str(i), "text": "Да?" if i < 2 else "Да.", "duration": 5.} for i in range(10)]
        repeats, report = even_repeats(rows, ["dialogs"] * 10, draws=83)
        self.assertEqual(sum(repeats), 83)
        self.assertEqual(report["question_draws"], 25)
        for pool in (repeats[:2], repeats[2:]):
            self.assertLessEqual(max(pool)-min(pool), 1)
        with self.assertRaisesRegex(ValueError, "30%"):
            even_repeats(rows, ["dialogs"] * 10, draws=10)

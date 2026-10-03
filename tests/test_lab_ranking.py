import tempfile
import unittest
from pathlib import Path

from newsrag.core import ValidationError
from newsrag.lab_ranking import (
    LinearFusionRanker,
    VerdictMapping,
    duplicate_rate,
    paired_bootstrap,
    ranking_metrics,
    ranks_from_scores,
    temporal_split,
)


def synthetic(n=60, k=6):
    import random
    rng = random.Random(3)
    queries = []
    for _ in range(n):
        rows = [[rng.gauss(0, 1), rng.gauss(0, 1)] for _ in range(k)]
        rows[0][0] += 2.5          # first feature separates the true candidate
        queries.append(rows)
    return queries


class RankingMetricsTests(unittest.TestCase):
    def test_ranks_ties_pessimistic_and_metrics(self):
        self.assertEqual(ranks_from_scores([[1, 1, 0]], [0]), [2])
        self.assertEqual(ranks_from_scores([[3, 1, 0], [0, 5, 1]], [0, 2]), [1, 2])
        m = ranking_metrics([1, 2, 6])
        self.assertAlmostEqual(m['mrr'], (1 + .5 + 1 / 6) / 3)
        self.assertAlmostEqual(m['top5'], 2 / 3)
        with self.assertRaises(ValidationError):
            ranks_from_scores([[1, 2]], [5])
        with self.assertRaises(ValidationError):
            ranking_metrics([0])

    def test_paired_bootstrap_detects_and_not_detects(self):
        a = [1.0] * 40
        b = [0.0] * 40
        self.assertTrue(paired_bootstrap(a, b).excludes_zero)
        same = [0.0, 1.0] * 20
        diff = paired_bootstrap(same, [1.0, 0.0] * 20)
        self.assertEqual(round(diff.mean, 6), 0)
        self.assertFalse(diff.excludes_zero)
        with self.assertRaises(ValidationError):
            paired_bootstrap([1.0], [1.0])


class SplitAndVerdictTests(unittest.TestCase):
    def test_temporal_split_and_gates(self):
        items = [{'d': '2022-05-01'}, {'d': '2023-01-01'}, {'d': '2024-02-02 10:00:00'}]
        train, test = temporal_split(items, lambda i: i['d'], '2023-01-01')
        self.assertEqual((len(train), len(test)), (1, 2))
        with self.assertRaises(ValidationError):
            temporal_split(items, lambda i: i['d'], '2030-01-01')
        with self.assertRaises(ValidationError):
            temporal_split([{'d': 'bad'}], lambda i: i['d'], '2023-01-01')

    def test_duplicate_rate_normalizes_punctuation(self):
        self.assertEqual(duplicate_rate(['مرحبا، بالعالم!'], ['مرحبا بالعالم', 'شيء آخر']), 0.5)

    def test_verdict_mapping_rejects_unknown_and_requires_caveat(self):
        m = VerdictMapping('x', {'مضلل': 'جزئي'}, 'لا تكافؤ مؤكد')
        self.assertEqual(m.normalize('مضلل'), 'جزئي')
        with self.assertRaises(ValidationError):
            m.normalize('زائف')
        with self.assertRaises(ValidationError):
            VerdictMapping('x', {'a': 'b'}, ' ')


class LinearFusionTests(unittest.TestCase):
    def test_trains_improves_and_saves_reloads(self):
        q = synthetic()
        ranker = LinearFusionRanker(['strong', 'noise'], feature_revision='r1')
        with self.assertRaises(ValidationError):
            ranker.scores([[0.0, 0.0]])
        hist = ranker.fit(q, label_source='weak_title_body', epochs=150)
        self.assertLess(hist[-1], hist[0])
        self.assertGreater(ranker.weights[0], ranker.weights[1])
        rows = synthetic(30)
        ranks = ranks_from_scores([ranker.scores(r) for r in rows], [0] * len(rows))
        self.assertGreater(ranking_metrics(ranks)['mrr'], 0.7)
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / 'r'
            ranker.save(target)
            loaded = LinearFusionRanker.load(target, feature_names=['strong', 'noise'],
                                             feature_revision='r1')
            self.assertEqual(loaded.scores(rows[0]), ranker.scores(rows[0]))
            self.assertEqual(loaded.label_sources, frozenset({'weak_title_body'}))
            with self.assertRaises(ValidationError):
                LinearFusionRanker.load(target, feature_names=['strong', 'noise'],
                                        feature_revision='other')
            with self.assertRaises(ValidationError):
                ranker.save(target)  # directory must be fresh
            (target / 'weights.npz').write_bytes((target / 'weights.npz').read_bytes() + b'x')
            with self.assertRaises(ValidationError):
                LinearFusionRanker.load(target, feature_names=['strong', 'noise'],
                                        feature_revision='r1')

    def test_rejects_bad_inputs_and_unknown_label_source(self):
        r = LinearFusionRanker(['a'], feature_revision='r')
        with self.assertRaises(ValidationError):
            r.fit([[[1.0], [2.0]]], label_source='synthetic_test_fixture')
        with self.assertRaises(ValidationError):
            r.fit([[[1.0, 2.0], [2.0, 3.0]]], label_source='weak_title_body')
        with self.assertRaises(ValidationError):
            r.fit([[[float('nan')], [1.0]]], label_source='weak_title_body')
        with self.assertRaises(ValidationError):
            LinearFusionRanker(['a', 'a'], feature_revision='r')
        with self.assertRaises(ValidationError):
            r.save('/tmp/should-not-exist-untrained')


if __name__ == '__main__':
    unittest.main()

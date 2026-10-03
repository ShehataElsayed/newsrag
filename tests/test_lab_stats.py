import unittest

from newsrag.core import ValidationError
from newsrag.lab_stats import (
    bootstrap_interval,
    calibration,
    error_breakdown,
    mcnemar_exact,
    minimum_detectable_difference,
    paired_permutation_pvalue,
    wilson_interval,
)


class StatsTests(unittest.TestCase):
    def test_bootstrap_and_wilson_contain_estimate(self):
        i = bootstrap_interval([0, 1] * 50)
        self.assertLess(i.low, 0.5)
        self.assertGreater(i.high, 0.5)
        w = wilson_interval(95, 100)
        self.assertLess(w.low, 0.95)
        self.assertTrue(0.88 < w.low < 0.92 and w.high < 0.99)
        with self.assertRaises(ValidationError):
            wilson_interval(5, 4)
        with self.assertRaises(ValidationError):
            bootstrap_interval([1.0])

    def test_mcnemar_exact_known_values(self):
        r = mcnemar_exact([True] * 10 + [False] * 2, [False] * 10 + [True] * 2)
        self.assertEqual((r['only_a'], r['only_b']), (10, 2))
        self.assertAlmostEqual(r['p_value'], 0.03857421875, places=9)
        self.assertEqual(mcnemar_exact([True, False], [True, False])['p_value'], 1.0)

    def test_permutation_detects_shift_not_noise(self):
        a = [1.0] * 30
        b = [0.0] * 30
        self.assertLess(paired_permutation_pvalue(a, b), 0.01)
        c = [0.0, 1.0] * 15
        d = [1.0, 0.0] * 15
        self.assertGreater(paired_permutation_pvalue(c, d), 0.5)

    def test_minimum_detectable_difference_shrinks_with_n(self):
        self.assertGreater(minimum_detectable_difference(80, .2),
                           minimum_detectable_difference(400, .2))

    def test_calibration_perfect_and_overconfident(self):
        perfect = calibration([1.0] * 10 + [0.0] * 10, [True] * 10 + [False] * 10)
        self.assertEqual((perfect.brier, perfect.ece), (0.0, 0.0))
        over = calibration([0.95] * 20, [True] * 10 + [False] * 10)
        self.assertAlmostEqual(over.ece, 0.45)
        with self.assertRaises(ValidationError):
            calibration([1.2], [True])

    def test_error_breakdown(self):
        r = error_breakdown(['a', 'a', 'b'], [True, False, True])
        self.assertEqual(r['a'], {'count': 2, 'accuracy': 0.5})
        with self.assertRaises(ValidationError):
            error_breakdown(['a'], [])


if __name__ == '__main__':
    unittest.main()

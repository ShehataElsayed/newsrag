import unittest

from newsrag.core import ValidationError
from newsrag.lab_drift import label_drift

C = ['a', 'b']


class DriftTests(unittest.TestCase):
    def test_identical_is_small(self) -> None:
        r = label_drift(['a'] * 50 + ['b'] * 50, ['a'] * 50 + ['b'] * 50, C)
        self.assertAlmostEqual(r.psi, 0.0)
        self.assertEqual(r.level, 'small')

    def test_shift_is_large(self) -> None:
        r = label_drift(['a'] * 90 + ['b'] * 10, ['a'] * 40 + ['b'] * 60, C)
        self.assertEqual(r.level, 'large')
        self.assertAlmostEqual(r.total_variation, 0.5, places=1)

    def test_validation(self) -> None:
        with self.assertRaises(ValidationError):
            label_drift([], ['a'], C)
        with self.assertRaises(ValidationError):
            label_drift(['a'], ['a'], ['a'])

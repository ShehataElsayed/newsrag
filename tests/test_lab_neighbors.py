import unittest

from newsrag.core import ValidationError
from newsrag.lab_neighbors import neighbour_gap


class NeighbourTests(unittest.TestCase):
    def test_pure_label_mix_standardises_to_zero(self) -> None:
        # near group is mostly class a, which is always right; novel has more b.
        sim = [0.9] * 10 + [0.1] * 10
        labels = ['a'] * 8 + ['b'] * 2 + ['a'] * 2 + ['b'] * 8
        correct = [lab == 'a' for lab in labels]
        g = neighbour_gap(sim, correct, labels, 0.5)
        self.assertGreater(g.raw_gap, 0.5)
        assert g.standardised_gap is not None
        self.assertAlmostEqual(g.standardised_gap, 0.0)

    def test_missing_class_is_dropped_with_coverage(self) -> None:
        g = neighbour_gap([0.9, 0.9, 0.1, 0.1], [True, True, True, False], ['a', 'a', 'a', 'b'], 0.5)
        assert g.standardised_gap is not None
        self.assertAlmostEqual(g.standardised_gap, 0.0)
        self.assertAlmostEqual(g.coverage, 0.5)

    def test_no_shared_class_gives_none(self) -> None:
        g = neighbour_gap([0.9, 0.1], [True, True], ['a', 'b'], 0.5)
        self.assertIsNone(g.standardised_gap)
        self.assertEqual(g.coverage, 0.0)

    def test_validation(self) -> None:
        with self.assertRaises(ValidationError):
            neighbour_gap([0.9], [True], ['a'], 0.5)
        with self.assertRaises(ValidationError):
            neighbour_gap([0.9], [True, True], ['a'], 0.5)

import unittest

from newsrag.core import ValidationError
from newsrag.lab_contradiction import contradiction_gate, gate_report


class GateTests(unittest.TestCase):
    def test_gate_flags_at_threshold(self) -> None:
        self.assertEqual(contradiction_gate([0.1, 0.5, 0.9], threshold=0.5), [False, True, True])

    def test_report_separates_kept_and_flagged(self) -> None:
        r = gate_report([False, False, True, True], [True, True, False, True])
        self.assertEqual((r.total, r.flagged), (4, 2))
        self.assertAlmostEqual(r.coverage, 0.5)
        assert r.accuracy_kept is not None and r.accuracy_flagged is not None
        self.assertAlmostEqual(r.accuracy_kept.estimate, 1.0)
        self.assertAlmostEqual(r.accuracy_flagged.estimate, 0.5)

    def test_nothing_flagged(self) -> None:
        r = gate_report([False, False], [True, False])
        self.assertIsNone(r.accuracy_flagged)

    def test_validation(self) -> None:
        with self.assertRaises(ValidationError):
            contradiction_gate([0.2], threshold=0.0)
        with self.assertRaises(ValidationError):
            contradiction_gate([1.2])
        with self.assertRaises(ValidationError):
            gate_report([], [])

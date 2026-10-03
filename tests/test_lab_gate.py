import unittest

import numpy as np

from newsrag.core import ValidationError
from newsrag.lab_gate import auroc, fit_agreement_gate


class GateTests(unittest.TestCase):
    def test_auroc_extremes_and_ties(self) -> None:
        self.assertAlmostEqual(auroc([0.1, 0.2, 0.8, 0.9], [False, False, True, True]), 1.0)
        self.assertAlmostEqual(auroc([0.9, 0.8, 0.2, 0.1], [False, False, True, True]), 0.0)
        self.assertAlmostEqual(auroc([0.5, 0.5, 0.5, 0.5], [False, True, False, True]), 0.5)

    def test_fit_learns_a_signal(self) -> None:
        rng = np.random.default_rng(0)
        x = rng.normal(size=(400, 3))
        y = (x[:, 0] + 0.2 * rng.normal(size=400)) > 0
        gate = fit_agreement_gate(x, y.tolist())
        p = gate.probability(x)
        self.assertGreater(auroc(p.tolist(), y.tolist()), 0.95)
        self.assertFalse(gate.status.endswith('True'))

    def test_validation(self) -> None:
        with self.assertRaises(ValidationError):
            auroc([0.1], [True])
        with self.assertRaises(ValidationError):
            fit_agreement_gate(np.zeros((20, 2)), [True] * 20)
        with self.assertRaises(ValidationError):
            fit_agreement_gate(np.zeros((5, 2)), [True, False] * 2 + [True])

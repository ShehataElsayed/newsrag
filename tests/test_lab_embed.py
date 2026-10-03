import unittest

import numpy as np

from newsrag.core import ValidationError
from newsrag.lab_embed import nearest_label_transfer


class TransferTests(unittest.TestCase):
    def test_transfer_and_majority(self) -> None:
        train = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
        test = np.array([[1.0, 0.0], [0.0, 1.0]])
        r = nearest_label_transfer(train, ['a', 'b', 'b'], test, ['a', 'a'])
        self.assertEqual(r.hits, [True, False])
        self.assertAlmostEqual(r.accuracy.estimate, 0.5)
        self.assertAlmostEqual(r.majority.estimate, 0.0)
        self.assertEqual(r.neighbour[0], 0)

    def test_validation(self) -> None:
        with self.assertRaises(ValidationError):
            nearest_label_transfer(np.zeros((1, 2)), ['a', 'b'], np.zeros((1, 2)), ['a'])

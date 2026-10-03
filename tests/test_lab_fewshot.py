import unittest

from newsrag.core import ValidationError
from newsrag.lab_fewshot import (
    build_fewshot_prompt,
    choice_probabilities,
    decide_fewshot,
)

C = ['False', 'Partly', 'True']


class FewShotTests(unittest.TestCase):
    def test_prompt_keeps_example_order_and_ends_open(self) -> None:
        prompt, letters = build_fewshot_prompt('new claim', [('first', 'Partly'), ('second', 'True')], C)
        self.assertEqual(letters, ['A', 'B', 'C'])
        self.assertLess(prompt.index('first'), prompt.index('second'))
        self.assertTrue(prompt.endswith('Claim: new claim\nVerdict:'))
        self.assertIn('Claim: first\nVerdict: B', prompt)

    def test_probabilities_sum_to_one(self) -> None:
        p = choice_probabilities([1.0, 2.0, 3.0])
        self.assertAlmostEqual(sum(p), 1.0)
        self.assertGreater(p[2], p[1])

    def test_decision(self) -> None:
        self.assertEqual(decide_fewshot([0.1, 0.7, 0.2], C, 0.6).suggestion, 'Partly')
        d = decide_fewshot([0.4, 0.4, 0.2], C, 0.6)
        self.assertTrue(d.abstain)
        self.assertFalse(d.validated_for_release)

    def test_validation(self) -> None:
        with self.assertRaises(ValidationError):
            build_fewshot_prompt('c', [('x', 'Unknown')], C)
        with self.assertRaises(ValidationError):
            decide_fewshot([0.5, 0.5], C, 0.6)
        with self.assertRaises(ValidationError):
            choice_probabilities([])

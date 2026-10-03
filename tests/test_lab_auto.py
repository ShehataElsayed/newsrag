import unittest

from newsrag.core import ValidationError
from newsrag.lab_auto import PipelineConfig, claim_type, decide


class AutoTests(unittest.TestCase):
    def test_claim_types(self) -> None:
        self.assertEqual(claim_type('unemployment fell 12%'), 'numeric')
        self.assertEqual(claim_type('the minister said it is safe'), 'attribution')
        self.assertEqual(claim_type('new vaccine is unsafe 5'), 'health')
        self.assertEqual(claim_type('a plain statement'), 'general')

    def test_low_similarity_abstains(self) -> None:
        d = decide('general', ['a'], [0.3], [0.0])
        self.assertTrue(d.abstain)
        self.assertEqual(d.reasons, ('low_similarity',))
        self.assertFalse(d.validated_for_release or d.autonomous_allowed)

    def test_contradiction_abstains_only_from_stage_3(self) -> None:
        self.assertEqual(decide('general', ['a'], [0.9], [0.9], stages=2).suggestion, 'a')
        d = decide('general', ['a'], [0.9], [0.9], stages=3)
        self.assertEqual(d.reasons, ('contradiction',))

    def test_vote_and_skip(self) -> None:
        split = decide('general', ['a', 'b'], [0.9, 0.9], [0.0, 0.0])
        self.assertEqual(split.reasons, ('split_vote',))
        ok = decide('general', ['a', 'a', 'b'], [0.9, 0.85, 0.82], [0.0] * 3)
        self.assertEqual(ok.suggestion, 'a')
        skip = decide('health', ['a'], [0.9], [0.0], PipelineConfig(skip_types=frozenset({'health'})))
        self.assertEqual(skip.reasons, ('skipped_type',))

    def test_validation(self) -> None:
        with self.assertRaises(ValidationError):
            decide('general', ['a'], [0.9, 0.8], [0.0])
        with self.assertRaises(ValidationError):
            decide('general', ['a'], [0.9], [0.0], stages=5)

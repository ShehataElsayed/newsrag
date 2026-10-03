import unittest

import numpy as np

from newsrag import ValidationError
from newsrag.neural_components import (
    ClaimStructureTokenHead,
    ClaimTokenHead,
    EvidenceContradictionHead,
    LabeledFeature,
    NegationScopeTokenHead,
    QuantityAlignmentHead,
    SourceDependenceHead,
    SpanTokenHead,
    StanceHead,
    TemporalRelationHead,
)


class TestTaskHeads(unittest.TestCase):
    def head(self, kind=StanceHead):
        return kind(input_width=3, hidden_size=4, seed=9,
                    feature_id='constructed_fixture', feature_revision='v1')

    def cases(self):
        return [LabeledFeature((1., 0., .2), 0, 'synthetic-a', 'synthetic_test_fixture'),
                LabeledFeature((0., 1., .3), 1, 'synthetic-b', 'synthetic_test_fixture')]

    def test_finite_difference_all_weight_families(self):
        h = self.head()
        x = np.array([c.features for c in self.cases()])
        y = np.array([0, 1])
        _, grads = h._loss_grad(x, y, .01)
        for name, index in [('w1', (0, 0)), ('b1', (0,)), ('w2', (0, 1)), ('b2', (1,))]:
            arr = getattr(h, name)
            original, epsilon = arr[index], 1e-6
            arr[index] = original + epsilon
            plus = h._loss_grad(x, y, .01)[0]
            arr[index] = original - epsilon
            minus = h._loss_grad(x, y, .01)[0]
            arr[index] = original
            self.assertAlmostEqual(grads[name][index], (plus-minus)/(2*epsilon), places=5)

    def test_three_heads_optimize_and_report_synthetic_provenance(self):
        for kind in (StanceHead, SpanTokenHead, SourceDependenceHead):
            h = self.head(kind)
            with self.assertRaises(ValidationError):
                h.class_scores([[1., 0., 0.]])
            history = h.fit(self.cases(), held_out_groups=frozenset({'unused-test'}),
                            epochs=200, learning_rate=.05)
            self.assertLess(history[-1], history[0])
            p = h.class_scores([c.features for c in self.cases()])
            self.assertEqual(p.shape, (2, len(h.labels)))
            np.testing.assert_allclose(p.sum(axis=1), 1)
            self.assertFalse(h.status()['human_reviewed_only'])
            self.assertFalse(h.status()['validated_for_release'])
            self.assertIsNone(h.status()['truth_probability'])

    def test_weak_label_provenance_is_not_human_review(self):
        h = self.head()
        cases = [LabeledFeature((1., 0., 0.), 0, 'doc-a', 'weak_surface_rule')]
        h.fit(cases, held_out_groups=frozenset({'doc-b'}), epochs=2)
        self.assertEqual(h.status()['label_sources'], ['weak_surface_rule'])
        self.assertFalse(h.status()['human_reviewed_only'])
        with self.assertRaises(ValidationError):
            LabeledFeature((1.,), 0, 'g', 'automatically_verified')

    def test_group_shape_label_and_nonfinite_gates(self):
        with self.assertRaises(ValidationError):
            self.head().fit(self.cases(), held_out_groups=frozenset({'synthetic-a'}))
        with self.assertRaises(ValidationError):
            self.head().fit([LabeledFeature((1.,), 0, 'a', 'weak_surface_rule')],
                            held_out_groups=frozenset({'b'}))
        with self.assertRaises(ValidationError):
            self.head(SpanTokenHead).fit([
                LabeledFeature((1., 0., 0.), 2, 'a', 'weak_surface_rule')],
                held_out_groups=frozenset({'b'}))
        with self.assertRaises(ValidationError):
            LabeledFeature((float('nan'),), 0, 'a', 'weak_surface_rule')
        h = self.head()
        h.fit(self.cases(), held_out_groups=frozenset({'unused-test'}), epochs=1)
        with self.assertRaises(ValidationError):
            h.class_scores([[float('inf'), 0., 0.]])

    def test_parameter_count_and_status(self):
        h = StanceHead(input_width=1542, hidden_size=64,
                       feature_id='pair', feature_revision='v1')
        self.assertEqual(h.parameter_count, 98947)
        self.assertFalse(h.status()['trained'])

    def test_remaining_six_heads_have_distinct_tasks_and_optimize(self):
        kinds = (ClaimTokenHead, ClaimStructureTokenHead, QuantityAlignmentHead,
                 TemporalRelationHead, NegationScopeTokenHead, EvidenceContradictionHead)
        self.assertEqual(len({k.task for k in kinds}), 6)
        for kind in kinds:
            h = self.head(kind)
            with self.assertRaises(ValidationError):
                h.class_scores([[1., 0., 0.]])
            history = h.fit(self.cases(), held_out_groups=frozenset({'unused-test'}),
                            epochs=80, learning_rate=.05)
            self.assertLess(history[-1], history[0])
            self.assertFalse(h.status()['validated_for_release'])
            self.assertEqual(h.status()['label_sources'], ['synthetic_test_fixture'])
            self.assertIsNone(h.status()['truth_probability'])
            np.testing.assert_allclose(h.class_scores([[1., 0., .2]]).sum(), 1.)

    def test_remaining_head_gradient_representatives(self):
        for kind in (ClaimStructureTokenHead, TemporalRelationHead, ClaimTokenHead):
            h = self.head(kind)
            x = np.array([c.features for c in self.cases()])
            y = np.array([0, 1])
            _, gradients = h._loss_grad(x, y, .01)
            for name, index in [('w1', (0, 0)), ('b1', (0,)),
                                ('w2', (0, 1)), ('b2', (1,))]:
                array = getattr(h, name)
                value, epsilon = array[index], 1e-6
                array[index] = value+epsilon
                plus = h._loss_grad(x, y, .01)[0]
                array[index] = value-epsilon
                minus = h._loss_grad(x, y, .01)[0]
                array[index] = value
                self.assertAlmostEqual(gradients[name][index], (plus-minus)/(2*epsilon), places=5)

    def test_external_professional_label_keeps_original_reference(self):
        with self.assertRaises(ValidationError):
            LabeledFeature((1., 0., 0.), 0, 'event-a', 'external_professional')
        h = self.head()
        h.fit([LabeledFeature((1., 0., 0.), 0, 'event-a', 'external_professional',
                              'https://example.test/professional-label-fixture')],
              held_out_groups=frozenset({'event-b'}), epochs=2)
        self.assertFalse(h.status()['human_reviewed_only'])
        self.assertEqual(h.status()['label_sources'], ['external_professional'])
        self.assertEqual(len(h.status()['label_references']), 1)

    def test_save_load_exact_scores_and_weak_provenance(self):
        import json
        import tempfile
        from pathlib import Path

        h = self.head()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'head'
            with self.assertRaises(ValidationError):
                h.save(path)
            h.fit(self.cases(), held_out_groups=frozenset({'unused-test'}), epochs=2)
            h.save(path)
            loaded = StanceHead.load(path, feature_id=h.feature_id,
                                      feature_revision=h.feature_revision)
            np.testing.assert_array_equal(h.class_scores([[1., 0., .2]]),
                                          loaded.class_scores([[1., 0., .2]]))
            self.assertEqual(h.status(), loaded.status())
            self.assertEqual((path/'weights.npz').stat().st_mode & 0o777, 0o600)
            with self.assertRaises(FileExistsError):
                h.save(path)
            with self.assertRaises(ValidationError):
                StanceHead.load(path, feature_id='different', feature_revision=h.feature_revision)
            with self.assertRaises(ValidationError):
                SpanTokenHead.load(path, feature_id=h.feature_id, feature_revision=h.feature_revision)
            metadata = json.loads((path/'metadata.json').read_text())
            metadata['validated_for_release'] = True
            (path/'metadata.json').write_text(json.dumps(metadata))
            with self.assertRaises(ValidationError):
                StanceHead.load(path, feature_id=h.feature_id, feature_revision=h.feature_revision)

    def test_corrupt_weights_rejected(self):
        import tempfile
        from pathlib import Path

        h = self.head()
        h.fit(self.cases(), held_out_groups=frozenset({'unused-test'}), epochs=2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'head'
            h.save(path)
            weights = path/'weights.npz'
            weights.write_bytes(weights.read_bytes()+b'corrupt')
            with self.assertRaises(ValidationError):
                StanceHead.load(path, feature_id=h.feature_id, feature_revision=h.feature_revision)

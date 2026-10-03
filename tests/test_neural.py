import tempfile
import unittest
from pathlib import Path

from newsrag import NewsroomRAG, Source, ValidationError
from newsrag.neural import (
    AbstentionGate,
    CalibrationCase,
    NeuralRanker,
    PairFeatures,
    ReviewedPair,
)


def synthetic_embed(texts):
    # Constructed vectors test code only, not Arabic model quality.
    return [[1., 0.] if 'صحة' in t else [0., 1.] for t in texts]


class TestExperimentalNeural(unittest.TestCase):
    def features(self):
        return PairFeatures(synthetic_embed, dimension=2, encoder_id='synthetic-only', encoder_revision='test-v1')

    def test_untrained_is_unusable(self):
        ranker = NeuralRanker(self.features())
        with self.assertRaises(ValidationError):
            ranker.scores('صحة', ())
        with tempfile.TemporaryDirectory() as d, self.assertRaises(ValidationError):
            ranker.save(Path(d)/'random')

    def test_dimensions_and_features(self):
        f = self.features()
        x = f.transform([('خبر صحة 20', 'صحة 20 اليوم')])
        self.assertEqual(x.shape, (1, 14))
        self.assertEqual(x[0, -4], 1.)  # numeric overlap
        f384 = PairFeatures(synthetic_embed, dimension=384, encoder_id='fake', encoder_revision='v1')
        self.assertEqual(NeuralRanker(f384).parameter_count, 98817)
        with self.assertRaises(ValidationError):
            f384.transform([('a', 'b')])

    def test_gradient_finite_difference(self):
        r = NeuralRanker(self.features(), hidden_size=3, seed=4)
        xp = r.features.transform([('صحة', 'صحة خبر')])
        xn = r.features.transform([('صحة', 'رياضة')])
        _, grad = r._loss_grad(xp, xn, .01)
        for name, index in [('w1', (0, 0)), ('w2', (0,)), ('b1', (0,))]:
            arr = getattr(r, name)
            old = arr[index]
            eps = 1e-6
            arr[index] = old+eps
            plus = r._loss_grad(xp, xn, .01)[0]
            arr[index] = old-eps
            minus = r._loss_grad(xp, xn, .01)[0]
            arr[index] = old
            self.assertAlmostEqual((plus-minus)/(2*eps), grad[name][index], places=5)

    def test_fit_adapter_and_private_save(self):
        r = NeuralRanker(self.features(), hidden_size=8, seed=3)
        pairs = [ReviewedPair('صحة', 'صحة خبر', 'رياضة', 'train-event', 'independently_reviewed')]
        history = r.fit(pairs, held_out_groups=frozenset({'test-event'}), epochs=100)
        self.assertLess(history[-1], history[0])
        core = NewsroomRAG().add(Source('a', 'A', 'صحة خبر'), Source('b', 'B', 'رياضة'))
        evidence = core.search('صحة رياضة')
        out = r('صحة', evidence)
        self.assertEqual(out[0].source_id, 'a')
        self.assertEqual({id(e) for e in out}, {id(e) for e in evidence})
        self.assertTrue(all(core._sources[e.source_id].text[e.start:e.end] == e.excerpt for e in out))
        with tempfile.TemporaryDirectory() as d:
            r.save(Path(d)/'model')
            self.assertEqual((Path(d)/'model/weights.npz').stat().st_mode & 0o777, 0o600)
            with self.assertRaises(FileExistsError):
                r.save(Path(d)/'model')

    def test_training_review_and_group_gates(self):
        with self.assertRaises(ValidationError):
            ReviewedPair('a', 'b', 'c', 'g', 'auto_generated')
        p = ReviewedPair('a', 'b', 'c', 'g', 'independently_reviewed')
        with self.assertRaises(ValidationError):
            NeuralRanker(self.features()).fit([p], held_out_groups=frozenset({'g'}))

    def test_abstention_calibration_disjoint_and_no_evidence(self):
        gate = AbstentionGate()
        with self.assertRaises(ValidationError):
            gate.accept([1.])
        cases = [CalibrationCase(2., True, 'cal-a', 'independently_reviewed'),
                 CalibrationCase(0., False, 'cal-b', 'independently_reviewed')]
        result = gate.calibrate(cases, training_groups=frozenset({'train'}), held_out_groups=frozenset({'test'}))
        self.assertEqual(result['held_out_performance'], None)
        self.assertTrue(gate.accept([2.]))
        self.assertFalse(gate.accept([0.]))
        self.assertFalse(gate.accept([]))
        with self.assertRaises(ValidationError):
            gate.calibrate(cases, training_groups=frozenset({'cal-a'}), held_out_groups=frozenset({'test'}))
        with self.assertRaises(ValidationError):
            gate.accept([float('nan')])

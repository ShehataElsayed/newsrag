import math
import unittest

from newsrag.core import ValidationError
from newsrag.lab_verdict import (
    AbstainingVerdict,
    HashedTfidf,
    choose_threshold,
    fit_temperature,
    normalize_claim,
    risk_coverage,
)

TRUE_T = ['فيديو يظهر فيضانا في المدينة الساحلية', 'صورة قديمة تنتشر على أنها حديثة']
FALSE_T = ['لا صحة لوفاة الفنان المشهور', 'خبر مفبرك عن قرار رسمي جديد']


def corpus():
    texts, labels = [], []
    for i in range(40):
        texts.append(f'{TRUE_T[i % 2]} رقم {i}'); labels.append('A')
        texts.append(f'{FALSE_T[i % 2]} رقم {i}'); labels.append('B')
    return texts, labels


def build(source='external_professional'):
    return AbstainingVerdict(['A', 'B'], label_source=source, platform='منصة اختبار',
                             equivalence_caveat='لا تكافؤ مع منصات أخرى', epochs=120, dim=512)


class VerdictTests(unittest.TestCase):
    def test_normalize_and_vectorizer(self):
        self.assertEqual(normalize_claim('أَخبارٌ  مُضلِّلة'), 'اخبار مضلله')
        v = HashedTfidf(256).fit(['كلمة واحدة', 'كلمة اخرى'])
        x = v.transform(['كلمة'])
        self.assertAlmostEqual(float((x ** 2).sum()), 1.0, places=4)
        with self.assertRaises(ValidationError):
            HashedTfidf(256).transform(['x'])

    def test_learns_and_always_defers_to_editor(self):
        texts, labels = corpus()
        m = build().fit(texts, labels).calibrate(texts, labels, target_accuracy=0.8, min_cases=10)
        d = m.decide(TRUE_T[0] + ' رقم 3')
        self.assertEqual(d.suggestion, 'A')
        self.assertFalse(d.validated_for_release)
        self.assertEqual(d.action, 'editor_review_required')
        self.assertFalse(m.autonomous_enabled)
        self.assertIn('منصة اختبار', d.notes[0])

    def test_autonomy_needs_reviewed_labels_and_volume(self):
        texts, labels = corpus()
        m = build('independently_reviewed').fit(texts, labels).calibrate(texts, labels, min_cases=10)
        self.assertFalse(m.autonomous_enabled)  # only 80 reviewed cases
        m.reviewed_cases = 500
        self.assertEqual(m.autonomous_enabled, math.isfinite(m.threshold))
        ext = build().fit(texts, labels).calibrate(texts, labels, min_cases=10)
        ext.reviewed_cases = 10_000
        self.assertFalse(ext.autonomous_enabled)

    def test_abstains_when_no_threshold_qualifies(self):
        self.assertEqual(choose_threshold([0.9] * 10, [True] * 10, target_accuracy=0.9, min_cases=30), math.inf)
        t = choose_threshold([0.9] * 100, [True] * 100, target_accuracy=0.9, min_cases=30)
        self.assertEqual(t, 0.9)
        texts, labels = corpus()
        m = build().fit(texts, labels)
        self.assertTrue(m.decide('نص عشوائي').abstained)  # uncalibrated threshold is infinite

    def test_temperature_and_risk_coverage(self):
        import numpy as np
        logits = np.array([[5.0, 0.0]] * 8 + [[0.0, 5.0]] * 2)
        t = fit_temperature(logits, [0] * 10)  # 20% of confident answers are wrong
        self.assertGreater(t, 1.0)
        rc = risk_coverage([0.9, 0.8, 0.7, 0.6], [True, True, False, True])
        self.assertEqual(rc[1][1:], (0.5, 1.0))
        self.assertAlmostEqual(rc[-1][2], 0.75)
        with self.assertRaises(ValidationError):
            risk_coverage([], [])

    def test_constructor_guards(self):
        with self.assertRaises(ValidationError):
            AbstainingVerdict(['A', 'B'], label_source='guess', platform='x', equivalence_caveat='y')
        with self.assertRaises(ValidationError):
            AbstainingVerdict(['A', 'B'], label_source='external_professional', platform='x', equivalence_caveat=' ')


if __name__ == '__main__':
    unittest.main()

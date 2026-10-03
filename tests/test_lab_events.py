import unittest

from newsrag.core import ValidationError
from newsrag.lab_events import (
    event_groups,
    group_kfold,
    leak_impact,
    nearest_train_similarity,
    random_kfold,
    split_overlap,
)

TEXTS = ['فيضان في مدينة جدة اليوم', 'فيضان في مدينة جدة اليوم مصور', 'مباراة كرة القدم النهائية غدا',
         'مباراة كرة القدم النهائية غدا بث', 'اكتشاف دواء جديد للسكري']


class EventTests(unittest.TestCase):
    def test_groups_join_near_duplicates_only(self):
        g = event_groups(TEXTS, threshold=0.5)
        self.assertEqual(g[0], g[1])
        self.assertEqual(g[2], g[3])
        self.assertEqual(len({g[0], g[2], g[4]}), 3)

    def test_group_kfold_keeps_groups_whole(self):
        groups = [i // 3 for i in range(60)]
        folds = group_kfold(groups, 5)
        self.assertEqual(split_overlap(groups, folds), 0)
        self.assertEqual(set(folds), {0, 1, 2, 3, 4})
        self.assertGreater(split_overlap(groups, random_kfold(60, 5)), 0)

    def test_nearest_similarity_and_leak_impact(self):
        sims = nearest_train_similarity(TEXTS[:2], [TEXTS[1], TEXTS[4]])
        self.assertGreater(sims[0], 0.9)
        self.assertLess(sims[1], 0.3)
        result = leak_impact([0.9] * 10 + [0.1] * 10, [True] * 10 + [True] * 5 + [False] * 5, threshold=0.6)
        self.assertEqual((result.near_count, result.novel_count), (10, 10))
        self.assertAlmostEqual(result.gap, 0.5)
        with self.assertRaises(ValidationError):
            leak_impact([0.9, 0.9], [True, False])
        with self.assertRaises(ValidationError):
            event_groups([])


if __name__ == '__main__':
    unittest.main()


class SemanticGroupingTests(unittest.TestCase):
    def test_vector_groups(self) -> None:
        from newsrag.lab_events import event_groups_from_vectors
        g = event_groups_from_vectors([[1, 0], [0.99, 0.05], [0, 1]], threshold=0.95)
        self.assertEqual(g[0], g[1])
        self.assertNotEqual(g[0], g[2])

    def test_shared_terms_respect_dates(self) -> None:
        from newsrag.lab_events import event_groups_shared_terms
        texts = ['flood hit Derna Libya dam collapse', 'Derna dam collapse video Libya flood old',
                 'cat football match unrelated story', 'Derna dam collapse images fake Libya',
                 'another story entirely about Cairo metro']
        g = event_groups_shared_terms(texts, [0, 3, 1, 5, 2], min_shared=3, max_doc_freq=0.9)
        self.assertEqual(g[0], g[1])
        self.assertEqual(g[0], g[3])
        self.assertNotEqual(g[2], g[0])
        far = event_groups_shared_terms(texts, [0, 400, 1, 800, 2], min_shared=3, max_doc_freq=0.9)
        self.assertNotEqual(far[0], far[1])

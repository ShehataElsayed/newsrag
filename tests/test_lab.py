import hashlib
import json
import unittest

from newsrag import ValidationError
from newsrag.lab import (
    EditorialDecision,
    EvidenceSpan,
    RawDocument,
    comparison_signals,
    exact_duplicate_groups,
    observed_numbers,
    review_export,
)


class TestEditorialControls(unittest.TestCase):
    def test_span_preserves_arabic_and_emoji_offsets(self):
        raw = '😀 قالت الوزارة: ارتفع العدد إلى ٢٠ شخصًا.'
        source = RawDocument('s', raw, 'https://example.test/s', 'research only')
        text = '٢٠ شخصًا'
        start = raw.index(text)
        span = EvidenceSpan('s', source.sha256, start, start + len(text), text)
        span.verify(source)
        self.assertEqual(raw[span.start:span.end], text)
        with self.assertRaises(ValidationError):
            EvidenceSpan('s', source.sha256, start + 1, start + 1 + len(text), text).verify(source)
        with self.assertRaises(ValidationError):
            EvidenceSpan('s', source.sha256, start, start+len(text), text,
                         'utf16_code_units').verify(source)

    def test_stale_source_and_wrong_id_rejected(self):
        a = RawDocument('a', 'خبر 20', 'https://example.test/a', 'research only')
        b = RawDocument('a', 'خبر 30', 'https://example.test/a', 'research only')
        span = EvidenceSpan('a', a.sha256, 0, len(a.text), a.text)
        with self.assertRaises(ValidationError):
            span.verify(b)
        with self.assertRaises(ValidationError):
            span.verify(RawDocument('b', a.text, a.source_url, a.rights))

    def test_duplicates_are_only_exact(self):
        docs = [RawDocument(str(i), text, 'https://example.test/'+str(i), 'research only')
                for i, text in enumerate(['نص', 'نص', 'نص آخر'])]
        self.assertEqual(exact_duplicate_groups(docs), (('0', '1'),))
        with self.assertRaises(ValidationError):
            exact_duplicate_groups([docs[0], docs[0]])

    def test_numeric_surface_and_negation_signals_not_verdicts(self):
        self.assertEqual(observed_numbers('النسبة ١٢٫٥٪ والعدد 20'), frozenset({'12.5%', '20'}))
        s = comparison_signals('ارتفع العدد إلى 20', 'لم يرتفع العدد إلى 30')
        self.assertEqual(s['claim_numbers_missing_from_excerpt'], ['20'])
        self.assertEqual(s['excerpt_negation_markers'], ['لم'])
        self.assertIsNone(s['automatic_verdict'])
        self.assertTrue(s['needs_editor_review'])
        same = comparison_signals('العدد 20', 'العدد 20')
        self.assertIsNone(same['automatic_verdict'])
        with self.assertRaises(ValidationError):
            comparison_signals('', 'خبر')

    def test_no_editor_verdict_without_reviewer_note_and_span(self):
        with self.assertRaises(ValidationError):
            EditorialDecision('خبر', 'editor_supported')
        with self.assertRaises(ValidationError):
            EditorialDecision('خبر', 'automatic_true')
        decision = EditorialDecision('خبر', 'editor_supported', 'reviewer-1', 'reviewed excerpt')
        with self.assertRaises(ValidationError):
            review_export(decision, [], [])
        self.assertIsNone(review_export(EditorialDecision('خبر', 'insufficient_evidence'), [], [])[
            'automatic_truth_probability'])

    def test_export_provenance_and_integrity(self):
        source = RawDocument('s', 'النص الخام', 'https://example.test/s', 'research only')
        span = EvidenceSpan('s', source.sha256, 0, len(source.text), source.text)
        exported = review_export(EditorialDecision('ادعاء', 'needs_review'), [source], [span])
        checksum = exported.pop('export_sha256')
        canonical = json.dumps(exported, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
        self.assertEqual(checksum, hashlib.sha256(canonical.encode()).hexdigest())
        self.assertEqual(exported['evidence'][0]['rights'], 'research only')
        with self.assertRaises(ValidationError):
            review_export(EditorialDecision('ادعاء', 'needs_review'), [], [span])

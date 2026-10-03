import unittest

from newsrag import ValidationError
from newsrag.lab import EditorialDecision, EvidenceSpan, RawDocument
from newsrag.lab_pipeline import (
    ProvenanceEdge,
    ProvenanceGraph,
    ReviewWorkspace,
    TokenOffset,
    decode_evidence_spans,
)


class TestLabPipeline(unittest.TestCase):
    def doc(self, name='a', text='قال 😀 عدد ١٢'):
        return RawDocument(name, text, f'https://example.test/{name}', 'fixture_only')

    def span(self, doc):
        return EvidenceSpan(doc.source_id, doc.sha256, 0, len(doc.text), doc.text)

    def test_decoder_preserves_arabic_and_emoji_raw_offsets(self):
        d = self.doc()
        offsets = [TokenOffset(0, 3, 'قال'), TokenOffset(4, 5, '😀'),
                   TokenOffset(6, 9, 'عدد'), TokenOffset(10, 12, '١٢')]
        spans = decode_evidence_spans(d, offsets, [0., 0., .9, .8], threshold=.7)
        self.assertEqual(len(spans), 1)
        self.assertEqual((spans[0].start, spans[0].end, spans[0].excerpt), (6, 12, 'عدد ١٢'))
        spans[0].verify(d)
        self.assertEqual(decode_evidence_spans(d, offsets, [0.]*4, threshold=.7), ())

    def test_decoder_does_not_join_omitted_content_or_unselected_tokens(self):
        d = self.doc(text='أ ب ج')
        offsets = [TokenOffset(0, 1, 'أ'), TokenOffset(4, 5, 'ج')]
        self.assertEqual(len(decode_evidence_spans(d, offsets, [.8, .8], threshold=.7)), 2)
        offsets.insert(1, TokenOffset(2, 3, 'ب'))
        self.assertEqual(len(decode_evidence_spans(d, offsets, [.8, .1, .8], threshold=.7)), 2)
        self.assertEqual(len(decode_evidence_spans(d, offsets, [.8]*3,
                                                 threshold=.7, max_chars=3)), 2)

    def test_decoder_rejects_invalid_offsets_even_unselected(self):
        d = self.doc()
        invalid = [TokenOffset(4, 6, '😀'), TokenOffset(True, 3, 'قال'),
                   TokenOffset(0, 3, 'قول')]
        for token in invalid:
            with self.assertRaises(ValidationError):
                decode_evidence_spans(d, [token], [0.], threshold=.7)
        with self.assertRaises(ValidationError):
            decode_evidence_spans(d, [TokenOffset(0, 3, 'قال')]*2, [.8]*2, threshold=.7)
        for scores, threshold in [([float('nan')], .7), ([1.2], .7), ([.8], -1.)]:
            with self.assertRaises(ValidationError):
                decode_evidence_spans(d, [TokenOffset(0, 3, 'قال')], scores,
                                      threshold=threshold)
        with self.assertRaises(ValidationError):
            decode_evidence_spans(d, [TokenOffset(0, 3, 'قال')], [], threshold=.7)

    def test_graph_transitive_cycle_safe_with_no_independence_claim(self):
        a, b, c = (self.doc(x) for x in 'abc')
        graph = ProvenanceGraph([a, b, c])
        graph.add(ProvenanceEdge('a', 'b', 'copies', 'fixture', 'test only', self.span(a)))
        graph.add(ProvenanceEdge('b', 'c', 'cites', 'fixture', 'test only', self.span(b)))
        self.assertEqual(graph.origin_path('a'), ('b', 'c'))
        self.assertFalse(graph.audit()['cycles_present'])
        graph.add(ProvenanceEdge('c', 'a', 'cites', 'fixture', 'test only', self.span(c)))
        self.assertTrue(graph.audit()['cycles_present'])
        self.assertIsNone(graph.audit()['independent_source_count'])
        self.assertFalse(graph.audit()['reviewer_identity_authenticated'])

    def test_graph_rejects_missing_stale_and_wrong_source_evidence(self):
        a, b = self.doc('a'), self.doc('b')
        graph = ProvenanceGraph([a, b])
        with self.assertRaises(ValidationError):
            graph.add(ProvenanceEdge('a', 'missing', 'copies', 'x', 'n', self.span(a)))
        with self.assertRaises(ValidationError):
            graph.add(ProvenanceEdge('a', 'b', 'copies', 'x', 'n', self.span(b)))
        with self.assertRaises(ValidationError):
            graph.add(ProvenanceEdge('a', 'b', 'copies', 'x', 'n', self.span(self.doc(text='غير'))))
        with self.assertRaises(ValidationError):
            ProvenanceEdge('a', 'b', 'copies', '', 'n', self.span(a))
        edge = ProvenanceEdge('a', 'b', 'copies', 'x', 'n', self.span(a))
        graph.add(edge)
        with self.assertRaises(ValidationError):
            graph.add(edge)
        with self.assertRaises(ValidationError):
            ProvenanceGraph([a, a])

    def test_workspace_selects_reviewed_raw_spans_not_model_verdict(self):
        w = ReviewWorkspace('عدد ١٢')
        d = self.doc()
        w.add_document(d)
        w.add_candidate(self.span(d))
        self.assertIsNone(w.diagnostics()[0]['automatic_verdict'])
        export = w.finish(EditorialDecision(w.claim, 'editor_supported',
                                           'fixture-reviewer', 'test-only review'), [0])
        self.assertEqual(export['evidence'][0]['excerpt'], d.text)
        self.assertIsNone(export['automatic_truth_probability'])
        with self.assertRaises(ValidationError):
            w.finish(EditorialDecision('غير', 'needs_review'), [0])
        with self.assertRaises(ValidationError):
            w.finish(EditorialDecision(w.claim, 'needs_review'), [0, 0])
        with self.assertRaises(ValidationError):
            w.finish(EditorialDecision(w.claim, 'needs_review'), [True])
        with self.assertRaises(ValidationError):
            w.add_document(d)
        with self.assertRaises(ValidationError):
            w.add_candidate(self.span(d))
        with self.assertRaises(ValidationError):
            w.add_candidate(self.span(self.doc('missing')))

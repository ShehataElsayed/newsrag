from unittest.mock import patch

from newsrag import ValidationError
from newsrag.collector import SourceSpec
from newsrag.orchestrate import run_investigation


def test_real_collector_failure_reaches_pipeline_as_gap_not_finding():
    """Offline protocol coverage: real collector and pipeline, failed network seam.
    Analysis seams frozen here, never an accuracy/model benchmark.
    """
    registry = [SourceSpec('البيان', ('https://publisher.test/report',), 'ناشر')]
    with patch('newsrag.collector._get', side_effect=ValidationError('simulated unavailable')), \
         patch('newsrag.orchestrate.decompose_question', return_value=[]), \
         patch('newsrag.orchestrate.extract_facets', return_value=None):
        result = run_investigation('كم مدرسة؟', registry,
                                    lambda prompt: (_ for _ in ()).throw(AssertionError('no model')))
    assert result['findings'] == []
    collection_gaps = [g for g in result['gaps'] if g['stage'] == 'collection']
    assert len(collection_gaps) == 1
    assert collection_gaps[0]['source'] == 'البيان'
    assert 'robots check failed' in collection_gaps[0]['reason']
    assert result['provenance'][0]['status'] == 'failed'
    assert result['provenance'][0]['raw_sha256'] is None


def test_policy_exclusion_retains_denial_without_network_or_findings():
    registry = [SourceSpec('مستبعد', ('https://ar.wikipedia.org/wiki/x',), 'ناشر')]
    with patch('newsrag.collector._get') as network, \
         patch('newsrag.orchestrate.decompose_question', return_value=[]), \
         patch('newsrag.orchestrate.extract_facets', return_value=None):
        result = run_investigation('سؤال؟', registry, lambda _: '')
    network.assert_not_called()
    assert not result['findings']
    assert result['provenance'][0]['status'] == 'denied'
    assert any(g['stage'] == 'collection' and 'denied' in g['reason'] for g in result['gaps'])

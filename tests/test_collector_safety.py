import hashlib
from unittest.mock import patch

import pytest

from newsrag import ValidationError
from newsrag.collector import SourceSpec, _denied, collect, fetch_url

URL = 'https://publisher.test/story'
ROBOTS = ('text/plain', b'User-agent: *\nAllow: /\n')
ARTICLE = '<article>أصلحت البلدية 12 مدرسة. لم تنشر تكلفة الإصلاح.</article>'.encode()


def test_pinned_transport_robots_order_and_hash_provenance():
    calls = []
    def get(host, path, timeout):
        calls.append((host, path))
        return ROBOTS if path == '/robots.txt' else ('text/html', ARTICLE)
    with patch('newsrag.collector._get', side_effect=get):
        result = collect('كم مدرسة؟', [SourceSpec('بيان', (URL,), 'بلدية')])
    assert calls == [('publisher.test', '/robots.txt'), ('publisher.test', '/story')]
    assert len(result['gathered']) == 1
    rec = result['provenance'][0]
    assert rec['raw_sha256'] == hashlib.sha256(ARTICLE).hexdigest()
    assert rec['text_sha256'] == hashlib.sha256(result['gathered'][0]['text'].encode()).hexdigest()
    assert rec['publisher'] == 'بلدية' and rec['speaker'] is None
    assert rec['transport'] == 'pinned_https'


@pytest.mark.parametrize('url', ['http://publisher.test/a', 'file:///etc/passwd',
                                   'https://user:pass@publisher.test/a',
                                   'https://publisher.test:444/a',
                                   'https://ar.wikipedia.org/wiki/x'])
def test_bad_urls_never_call_transport(url):
    with patch('newsrag.collector._get') as get, pytest.raises(ValidationError):
        fetch_url(url)
    get.assert_not_called()


def test_source_exclusion_matches_host_not_query_substring():
    assert _denied('https://AR.WIKIPEDIA.ORG./wiki/a')
    assert not _denied('https://publisher.test/?topic=wikipedia.org')
    assert not _denied('https://notwikipedia.org/a')


@pytest.mark.parametrize('robots', [b'User-agent: *\nDisallow: /\n',
                                   b'User-agent: *\nCrawl-delay: 5\n'])
def test_robots_denial_or_delay_never_fetch_article(robots):
    with patch('newsrag.collector._get', return_value=('text/plain', robots)) as get:
        result = collect('سؤال', [SourceSpec('بيان', (URL,), 'ناشر')])
    assert len(get.call_args_list) == 1
    assert not result['gathered']
    assert result['provenance'][0]['status'] == 'failed'


def test_empty_or_unsupported_content_not_evidence():
    for content_type, raw in [('text/html', b'<p> </p>'), ('image/png', b'not an article')]:
        with patch('newsrag.collector._get', side_effect=[ROBOTS, (content_type, raw)]):
            result = collect('سؤال', [SourceSpec('بيان', (URL,), 'ناشر')])
        assert not result['gathered']
        assert result['provenance'][0]['status'] == 'failed'


def test_redirect_failure_stays_gap_not_silence():
    with patch('newsrag.collector._get', side_effect=[ROBOTS, ValidationError('HTTP 302')]):
        result = collect('سؤال', [SourceSpec('بيان', (URL,), 'ناشر')])
    assert not result['gathered']
    assert 'article fetch failed' in result['provenance'][0]['detail']


def test_injected_transport_is_bounded_and_marked():
    class Response:
        def __init__(self, url): self.url = url
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def geturl(self): return self.url
        def read(self, limit):
            assert limit == 2_000_001
            return ROBOTS[1] if self.url.endswith('/robots.txt') else ARTICLE
    result = collect('سؤال', [SourceSpec('بيان', (URL,), 'ناشر')],
                     opener=lambda req, **kw: Response(req.full_url))
    assert result['provenance'][0]['transport'] == 'injected'
    assert result['gathered']


def test_injected_redirect_rejected():
    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def geturl(self): return 'https://ar.wikipedia.org/wiki/x'
    with pytest.raises(ValidationError):
        fetch_url(URL, opener=lambda *a, **kw: Response())

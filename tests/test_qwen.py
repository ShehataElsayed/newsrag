import json
import unittest
from typing import ClassVar
from urllib import request

from newsrag import NewsroomRAG, ValidationError
from newsrag.qwen import QwenProvider


def fake_transport(payload):
    def _t(req: request.Request):
        body = json.loads(req.data.decode("utf-8"))
        assert body["model"] == payload["expect_model"]
        assert body["messages"][0]["role"] == "user"
        return payload["response"]
    return _t


class QwenProviderTests(unittest.TestCase):
    OK: ClassVar = {"response": {"choices": [{"message": {"content": "خلاصة"}}]}, "expect_model": "qwen2.5:1.5b-instruct"}

    def test_local_default_needs_no_key_and_mock_never_counts_as_execution(self):
        p = QwenProvider(transport=fake_transport(self.OK))
        self.assertFalse(p.response_received)  # configured is not executed
        self.assertIsNone(p.model_reported)
        self.assertEqual(p.generate("اختبار"), "خلاصة")
        # injected/test transports must never flip execution flags
        self.assertFalse(p.response_received)
        self.assertIsNone(p.model_reported)

    def test_remote_requires_https(self):
        with self.assertRaises(ValidationError):
            QwenProvider(base_url="http://api.example.com/v1", api_key="k")

    def test_remote_requires_key(self):
        with self.assertRaises(ValidationError):
            QwenProvider(base_url="https://api.example.com/v1")

    def test_remote_https_with_key_sends_bearer(self):
        seen = {}
        def _t(req):
            seen["auth"] = req.headers.get("Authorization")
            return {"choices": [{"message": {"content": "ok"}}]}
        p = QwenProvider(base_url="https://api.example.com/v1", api_key="secret", transport=_t)
        p.generate("x")
        self.assertEqual(seen["auth"], "Bearer secret")

    def test_malformed_response_fails_closed(self):
        p = QwenProvider(transport=lambda req: {"choices": []})
        with self.assertRaises(ValidationError):
            p.generate("x")
        self.assertFalse(p.response_received)

    def test_empty_completion_fails_closed(self):
        p = QwenProvider(transport=lambda req: {"choices": [{"message": {"content": "  "}}]})
        with self.assertRaises(ValidationError):
            p.generate("x")
        self.assertFalse(p.response_received)

    def test_empty_model_rejected(self):
        with self.assertRaises(ValidationError):
            QwenProvider(model=" ")

    def test_plugs_into_newsroom_rag(self):
        rag = NewsroomRAG(generate=QwenProvider(transport=fake_transport(self.OK)).generate)
        self.assertEqual(rag.generate("سؤال"), "خلاصة")


class HostedProtocolTests(unittest.TestCase):
    def test_bounded_nonstreaming_payload(self):
        def transport(req):
            body = json.loads(req.data)
            self.assertIs(body['stream'], False)
            self.assertEqual(body['max_tokens'], 128)
            return {'choices': [{'message': {'content': '12'}, 'finish_reason': 'stop'}]}
        p = QwenProvider(max_tokens=128, transport=transport)
        self.assertEqual(p.generate('عدد المدارس؟'), '12')
        self.assertEqual(p.last_status, 'ok')

    def test_bad_endpoint_and_parameters(self):
        for url in ['ftp://localhost/v1', 'http://localhost.evil/v1',
                    'https://user:password@example.org/v1', 'https://example.org/v1?key=x',
                    'https://example.org/v1#x', 'https://example.org:bad/v1',
                    'https://example.org/with space']:
            with self.subTest(url=url), self.assertRaises(ValidationError):
                QwenProvider(base_url=url, api_key='k')
        for opts in [{'max_tokens': True}, {'max_tokens': 0}, {'max_tokens': 4097},
                     {'max_prompt_chars': 0}, {'timeout': float('nan')}, {'timeout': True}]:
            with self.subTest(opts=opts), self.assertRaises(ValidationError):
                QwenProvider(**opts)
        QwenProvider(base_url='http://[::1]:11434/v1')

    def test_invalid_prompt_never_calls_transport(self):
        p = QwenProvider(max_prompt_chars=4, transport=lambda req: self.fail('network call'))
        for prompt in ['', '  ', '12345', None]:
            with self.assertRaises(ValidationError):
                p.generate(prompt)
        self.assertEqual(p.last_status, 'rejected')

    def test_partial_refusal_tool_and_error_rejected(self):
        replies = [None, {'error': {'message': 'secret-bearing server error'}},
                   {'choices': [{'message': {'content': '12'}, 'finish_reason': 'length'}]},
                   {'choices': [{'message': {'content': '12', 'refusal': 'no'}}]},
                   {'choices': [{'message': {'content': '12', 'tool_calls': [{}]}}]},
                   {'choices': [{'message': {'content': '12'}, 'finish_reason': 'content_filter'}]}]
        for reply in replies:
            p = QwenProvider(transport=lambda req, r=reply: r)
            with self.subTest(reply=reply), self.assertRaises(ValidationError) as cm:
                p.generate('سؤال')
            self.assertNotIn('secret-bearing', str(cm.exception))
            self.assertEqual(p.last_status, 'failed')

    def test_metadata_allowlist_and_state_reset(self):
        calls = []
        def transport(req):
            calls.append(req)
            if len(calls) > 1:
                raise TimeoutError('timeout')
            return {'model': 'reported-only', 'usage': {'total_tokens': 12, 'neurons': 1.2,
                    'prompt': 'private', 'prompt_tokens': True, 'completion_tokens': float('nan')},
                    'choices': [{'message': {'content': 'خلاصة'}, 'finish_reason': 'stop'}]}
        p = QwenProvider(transport=transport)
        p.generate('سؤال')
        self.assertEqual(p.usage, {'total_tokens': 12, 'neurons': 1.2})
        self.assertIsNone(p.model_reported)  # mock never proves model execution
        with self.assertRaises(TimeoutError):
            p.generate('ثان')
        self.assertEqual(p.usage, {})
        self.assertIsNone(p.finish_reason)
        self.assertEqual(p.last_status, 'failed')
        self.assertEqual(len(calls), 2)  # no retries

    def test_no_redirect_or_unbounded_read(self):
        from unittest.mock import patch
        class Response:
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def read(self, limit):
                self.limit = limit
                return b'x' * limit
        response = Response()
        class Opener:
            def open(self, req, timeout): return response
        with patch('newsrag.qwen.request.build_opener', return_value=Opener()) as build:
            p = QwenProvider()
            with self.assertRaises(ValidationError):
                p.generate('question')
            self.assertEqual(response.limit, 2_000_001)
            handler = build.call_args.args[0]
            self.assertIsNone(handler.redirect_request(None, None, 302, '', {}, 'https://other.test'))

    def test_grounded_arabic_source_remains_exact_through_rag(self):
        from newsrag import Source
        source_text = 'أصلحت البلدية 12 مدرسة. لم تذكر تكلفة الإصلاح.'
        seen = []
        def transport(req):
            seen.append(json.loads(req.data)['messages'][0]['content'])
            return {'choices': [{'message': {'content': 'أصلحت البلدية 12 مدرسة [E1].'},
                                 'finish_reason': 'stop'}]}
        rag = NewsroomRAG(generate=QwenProvider(transport=transport).generate)
        rag.add(Source('fixture', 'بيان اصطناعي', source_text))
        result = rag.ask('كم مدرسة أصلحت البلدية؟')
        self.assertTrue(result.evidence)
        self.assertEqual(result.evidence[0].excerpt, source_text)
        self.assertTrue(result.citation_refs_valid)
        self.assertTrue(seen)
        # Citation validity is reference integrity, not semantic truth acceptance.

    def test_explicit_strict_completion_mode(self):
        p = QwenProvider(require_stop=True, transport=lambda req: {
            'choices': [{'message': {'content': 'partial?'}}]})
        with self.assertRaises(ValidationError):
            p.generate('سؤال')
        self.assertEqual(p.last_status, 'failed')
        p = QwenProvider(require_stop=True, transport=lambda req: {
            'choices': [{'message': {'content': '12'}, 'finish_reason': 'stop'}]})
        self.assertEqual(p.generate('سؤال'), '12')

    def test_llamaserver_locality_not_inferred(self):
        from newsrag.qwen import llamacpp_server_generate
        for url in ['http://example.org', 'https://example.org', 'file:///tmp/model',
                    'http://localhost.evil', 'http://user:pass@localhost']:
            with self.assertRaises(ValidationError):
                llamacpp_server_generate(base_url=url)
        llamacpp_server_generate(base_url='http://[::1]:8080')
        llamacpp_server_generate(base_url='https://explicit.example', allow_remote=True)
        with self.assertRaises(ValidationError):
            llamacpp_server_generate(base_url='http://explicit.example', allow_remote=True)

    def test_llamaserver_redirect_statuses_never_forward_prompt(self):
        import threading
        from http.server import BaseHTTPRequestHandler, HTTPServer
        from urllib.error import HTTPError

        from newsrag.qwen import llamacpp_server_generate
        forwarded = []
        class Target(BaseHTTPRequestHandler):
            def do_GET(self):
                forwarded.append('GET')
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b'{"content":"should not arrive"}')
            def do_POST(self):
                forwarded.append('POST')
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b'{"content":"should not arrive"}')
            def log_message(self, *args): pass
        target = HTTPServer(('127.0.0.1', 0), Target)
        status = [302]
        class Redirect(BaseHTTPRequestHandler):
            def do_POST(self):
                self.rfile.read(int(self.headers.get('Content-Length', 0)))
                self.send_response(status[0])
                self.send_header('Location', f'http://127.0.0.1:{target.server_port}/cross-boundary')
                self.end_headers()
            def log_message(self, *args): pass
        origin = HTTPServer(('127.0.0.1', 0), Redirect)
        threads = [threading.Thread(target=s.serve_forever, daemon=True) for s in (target, origin)]
        for t in threads: t.start()
        try:
            generate = llamacpp_server_generate(base_url=f'http://127.0.0.1:{origin.server_port}', timeout=2)
            for code in (301, 302, 303, 307, 308):
                status[0] = code
                with self.subTest(code=code), self.assertRaises(HTTPError):
                    generate('synthetic local-only prompt')
            self.assertEqual(forwarded, [])
        finally:
            for server in (origin, target):
                server.shutdown()
                server.server_close()
            for thread in threads: thread.join(timeout=2)

    def test_llamaserver_response_read_is_bounded(self):
        from unittest.mock import patch

        from newsrag.qwen import llamacpp_server_generate
        reads = []
        class Response:
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def read(self, limit):
                reads.append(limit)
                return b'x' * limit
        class Opener:
            def open(self, req, timeout): return Response()
        with patch('newsrag.qwen.request.build_opener', return_value=Opener()), self.assertRaises(ValidationError):
            llamacpp_server_generate()('synthetic prompt')
        self.assertEqual(reads, [2_000_001])

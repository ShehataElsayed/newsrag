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

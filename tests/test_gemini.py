import json
import unittest

from newsrag import GeminiProvider, NewsroomRAG, Source, ValidationError


def stub_transport(routes):
    def transport(req):
        action = req.full_url.rsplit(":", 1)[-1]
        payload = json.loads(req.data.decode("utf-8"))
        return routes[action](payload, req)
    return transport


class GeminiProviderTest(unittest.TestCase):
    def test_requires_key_and_model_names(self):
        for kwargs in ({"api_key": ""}, {"api_key": "  "},
                       {"api_key": "k", "model": ""}, {"api_key": "k", "embedding_model": ""}):
            with self.assertRaises(ValidationError):
                GeminiProvider(**kwargs)

    def test_generate_posts_prompt_with_key_header(self):
        seen = {}

        def gen(payload, req):
            seen["payload"] = payload
            seen["key"] = req.headers.get("X-goog-api-key")
            return {"candidates": [{"content": {"parts": [{"text": "مرحبا "}, {"text": "بالعالم"}]}}]}

        provider = GeminiProvider("SECRET", transport=stub_transport({"generateContent": gen}))
        self.assertEqual(provider.generate("قل مرحبا"), "مرحبا بالعالم")
        self.assertEqual(seen["key"], "SECRET")
        self.assertEqual(seen["payload"]["contents"][0]["parts"][0]["text"], "قل مرحبا")
        self.assertNotIn("SECRET", repr(seen["payload"]))

    def test_generate_rejects_bad_shapes(self):
        for body in ({}, {"candidates": []}, {"candidates": [{"content": {"parts": [{"text": "  "}]}}]},
                     {"candidates": [{"content": {}}]}):
            provider = GeminiProvider("k", transport=stub_transport({"generateContent": lambda p, r, b=body: b}))
            with self.assertRaises(ValidationError):
                provider.generate("س")

    def test_embed_returns_vectors_and_validates_count(self):
        def embed(payload, req):
            return {"embeddings": [{"values": [0.1, 0.2]} for _ in payload["requests"]]}

        provider = GeminiProvider("k", transport=stub_transport({"batchEmbedContents": embed}))
        vectors = provider.embed(["أ", "ب"])
        self.assertEqual(vectors, [[0.1, 0.2], [0.1, 0.2]])
        self.assertEqual(provider.embed([]), [])

        def short(payload, req):
            return {"embeddings": [{"values": [0.1]}]}

        bad = GeminiProvider("k", transport=stub_transport({"batchEmbedContents": short}))
        with self.assertRaises(ValidationError):
            bad.embed(["أ", "ب"])

    def test_embed_rejects_non_numeric(self):
        bad = GeminiProvider("k", transport=stub_transport(
            {"batchEmbedContents": lambda p, r: {"embeddings": [{"values": ["x"]}]}}))
        with self.assertRaises(ValidationError):
            bad.embed(["أ"])

    def test_end_to_end_with_newsroomrag(self):
        def gen(payload, req):
            return {"candidates": [{"content": {"parts": [{"text": "بدأ البرنامج التجريبي في سبتمبر [E1]."}]}}]}

        def embed(payload, req):
            return {"embeddings": [{"values": [1.0, 0.0]} for _ in payload["requests"]]}

        provider = GeminiProvider("k", transport=stub_transport(
            {"generateContent": gen, "batchEmbedContents": embed}))
        rag = NewsroomRAG(embed=provider.embed, generate=provider.generate).add(
            Source(id="1", title="بيان", text="بدأ برنامج تجريبي محدود في سبتمبر. لم يُعلن عن إطلاق عام."))
        answer = rag.ask("هل أُعلن عن إطلاق عام؟")
        self.assertTrue(answer.citation_refs_valid)
        self.assertEqual(answer.cited, ("E1",))
        self.assertEqual(answer.evidence[0].start, 0)


if __name__ == "__main__":
    unittest.main()

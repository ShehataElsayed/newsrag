import unittest

from newsrag import ArabicLightTokenizer, NewsroomRAG, Source


class FakeStemmer:
    def stemWord(self, token):
        return "مدرس" if token in {"بالمدارس", "المدرسة"} else token


class TestArabicTokenizer(unittest.TestCase):
    def test_surface_plus_stems(self):
        tokenizer = ArabicLightTokenizer.__new__(ArabicLightTokenizer)
        tokenizer.stemmer = FakeStemmer()
        self.assertEqual(tokenizer("بالمدارس"), ["بالمدارس", "مدرس"])
        rag = NewsroomRAG(tokenize=tokenizer).add(Source("a", "A", "الأطفال بالمدارس"))
        self.assertEqual(rag.search("المدرسة")[0].source_id, "a")

    def test_english_not_stemmed(self):
        tokenizer = ArabicLightTokenizer.__new__(ArabicLightTokenizer)
        tokenizer.stemmer = FakeStemmer()
        self.assertEqual(tokenizer("School report"), ["school", "report"])

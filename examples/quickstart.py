from datetime import datetime, timezone

from newsrag import NewsroomRAG, Source

rag = NewsroomRAG(recency_half_life_days=30).add(
    Source(id="press-release", title="Official announcement", text="The company announced a pilot on 10 September. It did not announce a public launch.", url="https://example.org/official", publisher="Example Corp", published_at=datetime(2026, 9, 10, tzinfo=timezone.utc)),
    Source(id="interview", title="Reporter notes", text="In an interview, the editor said the pilot still needs independent testing.", source_type="interview"),
)
print(rag.ask("Was there a public launch?").as_dict())
# Plug in ANY synchronous provider with two callables:
# rag = NewsroomRAG(embed=lambda texts: your_embedding_client(texts),
#                   generate=lambda prompt: your_llm_client(prompt))
# print(rag.add(...).ask("What changed? [in Arabic or English]").as_dict())

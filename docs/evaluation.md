# Evaluate retrieval on labeled questions

This small harness measures whether manually labeled source IDs appear in the first `k` distinct sources. It reports recall@k, hit rate@k, mean reciprocal rank, and missed queries. Labels need to be made by a human independently of the model's ranking. Keep an evaluation set separate from tuning. These scores say nothing about whether generated prose is true, citations support claims, sources are credible, or the dataset represents a real newsroom.

```python
from newsrag import NewsroomRAG, Source, RetrievalCase, evaluate_retrieval

rag = NewsroomRAG().add(
    Source("original", "Original statement", "The ministry announced a pilot."),
    Source("followup", "Follow-up", "The pilot was extended in October."),
)
cases = [RetrievalCase("What was announced?", frozenset({"original"})),
         RetrievalCase("What happened in October?", frozenset({"followup"}))]
print(evaluate_retrieval(rag, cases, top_k=2).as_dict())
```

Use source IDs that exist in the corpus; missing IDs count as misses. An embedding or reranking adapter configured on the RAG object is used during scoring and may make external calls, including transmitting queries and source text. Do not put confidential test material in an external provider without permission. Compare against an independent held-out set, track failures by topic and language, and inspect original sources before drawing editorial conclusions.

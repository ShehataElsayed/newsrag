# Architecture and contract

`Source` is an immutable record with required ID, title and text. Optional metadata includes original URL, publisher, source type, language, timezone-aware publication and access timestamps. Journalists should keep the source document separately; `Source` metadata is a claim about provenance, not a signature proving authorship.

`NewsroomRAG.add` chunks the text and keeps exact Python character offsets. It tokenizes basic Arabic and English, then scores matching chunks with a BM25-style lexical score. An optional caller-provided embedding function provides vectors for cosine ranking. A bounded, optional recency boost uses the publication date only; an unknown publication date is never filled in. `search` returns top passages with references (`E1`, `E2`) assigned per request. References are not durable IDs; persist the source ID, URL and offsets for an audit trail.

`ask` retrieves first. With no generator it returns passages only. With a generator it sends excerpts and citation instructions to the caller's model and checks whether bracketed reference numbers in the returned text exist in the retrieved set. It does not decide whether claims follow from evidence. A reviewer must open the original source, check date and context, inspect the quote and get independent corroboration for consequential claims.

`source_from_html` accepts already-fetched HTML and strips basic script/style content. It does not fetch or verify websites. `source_from_pdf` extracts text from caller-supplied bytes using optional pypdf; it is not OCR. `save_sources` stores original text and metadata as JSON, not embeddings. `load_sources` recomputes embeddings when given a configured rag instance. Storage is local and unencrypted.

## Provider adapter shape

```python
from newsrag import NewsroomRAG

rag = NewsroomRAG(
    embed=lambda texts: embedding_api(texts),  # list[str] -> list[list[float]]
    generate=lambda prompt: generation_api(prompt),  # str -> str
)
```

A provider must return one vector per input and use stable vector dimensions. Avoid changing the embedding model for a partially indexed corpus; build a fresh index if the model changes. Calls are synchronous and exceptions from providers propagate. Do not send source text to a provider without a valid editorial and privacy basis.

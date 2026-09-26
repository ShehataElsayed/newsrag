# NewsRAG by Shehata El-sayed

A small, auditable Python RAG toolkit for journalistic research. This v0.3 pre-release is a research prototype; check PyPI for publication status. Python 3.10+; standard library core. Licensed under Apache-2.0 (see LICENSE and NOTICE).

## Example

```python
from datetime import datetime, timezone
from newsrag import NewsroomRAG, Source

rag = NewsroomRAG(recency_half_life_days=30).add(
    Source(id="1", title="Official statement", text="A limited pilot began in September. No public launch was announced.",
           url="https://example.org/statement", published_at=datetime(2026, 9, 10, tzinfo=timezone.utc))
)
print(rag.ask("Was there a public launch?").as_dict())
```

Save a source's actual URL and publication date. The default `ask` retrieves evidence without generating prose. Add provider functions when you want answer generation:

```python
rag = NewsroomRAG(
    embed=lambda texts: embedding_provider(texts),   # list[str] -> list[list[float]]
    generate=lambda prompt: llm_provider(prompt),    # str -> str
).add(...)
answer = rag.ask("ما الذي أُعلن؟")
print(answer.text, answer.evidence, answer.warnings)
```

Install locally with `python -m pip install .`; run tests with `PYTHONPATH=src python -m unittest discover -s tests -v`, and run `PYTHONPATH=src python examples/quickstart.py` from this directory. No network calls occur unless your adapters make them.

## Newsroom-specific controls

- Every source carries an ID, title, original URL, publisher, type, timestamp with timezone, and original text. Search results retain exact offsets and excerpts for audit.
- Arabic orthographic normalization and English token search work locally. Optional provider embeddings improve semantic search and cross-language matching, but an appropriate multilingual model is required for Arabic-English semantic retrieval.
- Optional publication date range filters (`before`, `after`) and a modest recency ranking boost. Unknown publication dates are excluded when filtering by date; they remain eligible otherwise. The timestamp means publication time, not event time.
- Generator prompt asks for bracketed evidence IDs (`[E1]`). IDs are checked against retrieved passages and unknown IDs flagged. This does **not** verify the claims, the source's truth, or whether cited passages support each sentence.
- No-match answers abstain. Source text is marked untrusted, but prompt injection remains a risk with any model. Do not feed confidential material to a third-party provider without permission. Review original sources, licensing, quotes, dates, and contested claims before publishing.

## Scope and limitations

This v0.3 is an in-memory retrieval prototype with local JSON source snapshots, not a production crawler or fact-checker. It handles plain text and synchronous callable adapters; automatic fetching of URLs, OCR for scanned PDFs, database connectors, persistent vector indexes, source deduplication, named provider packages, automated entailment checking, and benchmark-based tuning are future work. Lexical Arabic tokenization is basic, not morphological. No claim that this works with every provider out of the box: providers must offer embeddings with stable vector dimensions and/or text generation and be wrapped in the two simple callables. The toy example.org URLs are not real news sources.


## Quality checks

Run `python -m pip install -e . pytest ruff mypy build "pypdf>=5,<7"`, then `ruff check src tests examples setup.py`, `mypy src/newsrag`, `pytest -q`, and `python -m build`. CI is configured to run these checks on Python 3.10–3.13; this local package was checked on Python 3.10 only. A passing suite reduces risk but cannot guarantee zero bugs or factual accuracy.

## Local ingestion and snapshots

```python
from newsrag import source_from_html, source_from_pdf, save_sources, load_sources

# You fetch the original page yourself. This library does not fetch URLs or bypass site terms.
source = source_from_html(id="story-1", title="Original article", url="https://example.org/story",
                          html="<article><p>An independently confirmed report.</p></article>")
rag.add(source)
save_sources(rag, "research.json")  # source content, mode 0600; no API keys or model code
restored = load_sources("research.json")
# PDF extraction, optional: pip install 'newsrag[pdf]'
# document = source_from_pdf(id="file-1", title="Report", pdf_bytes=pdf_data)
```

For web ingestion, the caller is responsible for fetching, verifying the URL, handling redirects, authentication, robots/licensing rules and SSRF protection. Passing HTML does not prove its authorship. HTML extraction is simple and may include navigation text; review the output. PDF extraction is text-only and may require OCR for scans. Snapshots store unencrypted source text, so keep them in a secure location; the writer uses 0600 permissions on POSIX. There is no network fetcher or hosted persistence. The latest source content should be rechecked at the original publisher before publication.

Further reading: [architecture and provider contract](docs/architecture.md), [release checklist](docs/release-checklist.md), [contributing](CONTRIBUTING.md), [security](SECURITY.md), and [changelog](CHANGELOG.md). These are preparation for a public release, not evidence it has been published or reviewed independently.

An optional provider illustration is in [`examples/openai_adapter.py`](examples/openai_adapter.py). It requires a separately installed SDK and the user's own provider credentials; no credentials belong in this repository. Model names and SDK behavior can change, so validate an adapter against its provider's current documentation before using it.

For release mechanics, see [PyPI Trusted Publishing setup](docs/publishing.md). No API token is needed in GitHub Actions. A GitHub release or a pending publisher alone is not proof that PyPI published the package; check the PyPI project and install before relying on it.

### Optional evidence reranking

Pass `rerank=lambda query, evidence: ...` when you have a trusted reranking function. The function receives the initially retrieved evidence and must return a permutation of those same objects; new or rewritten passages are rejected. Citation IDs are reassigned after the new order. Reranking cannot recover evidence absent from the initial `top_k`, and does not check whether an answer is true. Arabic normalization includes common hamza, ta marbuta, alef maqsura and Persian keyboard letter variants; it is still not Arabic stemming or morphological analysis.

### Retrieval evaluation

Use `RetrievalCase` and `evaluate_retrieval` with independently labeled relevant source IDs to measure recall@k, hit rate@k, and reciprocal rank. See [evaluation guide](docs/evaluation.md). These metrics do not validate generated claims or source credibility.

### Awaitable façade

`AsyncNewsroomRAG(NewsroomRAG(...))` exposes `await add(...)`, `replace(...)`, `remove(...)`, `search(...)` and `ask(...)`. It runs the synchronous core and its provider callables in worker threads and serializes operations on that instance even if an awaiting task is cancelled; cancellation does not stop provider work already running. This can keep an event loop responsive, but it is not a native coroutine provider interface and does not promise cross-process or direct-core thread safety. Do not mutate the wrapped `NewsroomRAG` outside the façade while tasks run.

### SQLite source store (v0.3)

```python
from newsrag import SQLiteSourceStore, NewsroomRAG, Source
with SQLiteSourceStore("research.db") as store:
    store.upsert(Source("story-1", "Story", "Verified source text goes here"))
    rag = NewsroomRAG().add(*store.sources())
```

SQLite stores source text and metadata transactionally and creates a new database with mode 0600 on POSIX. It does not persist embeddings or an already-built search index: rebuild chunks and vectors at startup, with provider costs if configured. The file is not encrypted; protect and back it up with SQLite's backup API rather than copying it while live. Do not use this as a multiwriter service without an application-level concurrency plan.

To make a consistent backup while the database is open, use `store.backup("research-backup.db")`. It refuses to overwrite an existing destination and creates the backup with mode 0600 on POSIX. Protect both files: neither is encrypted.

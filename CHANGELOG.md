# Changelog

## 0.3.0 - 2026-09-26

- Normalize common Arabic letter variants after Unicode NFKC normalization (including Persian keyboard variants).
- Allow an optional caller-supplied reranker to reorder retrieved evidence without changing or injecting passages; citation IDs are reassigned after reranking.
- Add tests for Arabic variants, reranking, citation IDs and rejected adapter output.
- Add a small deterministic retrieval evaluation harness with source-level recall@k, hit rate@k, mean reciprocal rank and missed queries; document its limits.
- Reject non-finite ranking configuration, nonnumeric embedding values and invalid source metadata with ValidationError.
- Add an awaitable façade that runs synchronous provider calls in worker threads and serializes operations per instance; it is not a native async provider contract.
- Keep the wrapped index serialized when a caller cancels a running worker-thread operation; cancellation does not stop provider work already underway.
- Add `remove(source_id)` in the sync core and awaitable façade for explicit source deletion.
- Add a transactional SQLite source store with restricted POSIX file mode. It persists source text and metadata, not a vector or lexical index.
- Add a consistent SQLite backup method that refuses overwrites and writes with restricted POSIX permissions.
- Add optional local multilingual sentence embeddings, normalized lexical/semantic score fusion, and optional Snowball Arabic light stemming.
- Add optional allowlisted single-URL HTTPS article fetching with public-DNS checks, robots permission, redirect rejection, response limits and Trafilatura extraction; no general crawler.

## 0.2.0 - 2026-09-26

- Added caller-supplied HTML and optional pypdf text extraction.
- Added local JSON source snapshots with atomic writes and restricted POSIX permissions.
- Expanded automated checks and source metadata documentation.

## 0.1.0 - 2026-09-26

- Initial provider-neutral in-memory retrieval with Arabic and English lexical matching.
- Added source metadata, exact offsets, date filters, optional embeddings and generation adapters.
- Added citation reference validation, abstention on no matches and review warnings.

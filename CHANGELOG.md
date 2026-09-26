# Changelog

## 0.2.0 - 2026-09-26

- Added caller-supplied HTML and optional pypdf text extraction.
- Added local JSON source snapshots with atomic writes and restricted POSIX permissions.
- Expanded automated checks and source metadata documentation.

## 0.1.0 - 2026-09-26

- Initial provider-neutral in-memory retrieval with Arabic and English lexical matching.
- Added source metadata, exact offsets, date filters, optional embeddings and generation adapters.
- Added citation reference validation, abstention on no matches and review warnings.

## Unreleased

- Normalize common Arabic letter variants after Unicode NFKC normalization (including Persian keyboard variants).
- Allow an optional caller-supplied reranker to reorder retrieved evidence without changing or injecting passages; citation IDs are reassigned after reranking.
- Add tests for Arabic variants, reranking, citation IDs and rejected adapter output.
- Add a small deterministic retrieval evaluation harness with source-level recall@k, hit rate@k, mean reciprocal rank and missed queries; document its limits.
- Reject non-finite ranking configuration, nonnumeric embedding values and invalid source metadata with ValidationError.

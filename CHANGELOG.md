- assist(): add the missing numeric support guard. Every number in a claim must
  appear (after Eastern/Western Arabic digit normalization) in its supporting
  sentence; a mismatch now fails closed toward extractive/insufficient instead
  of showing a number-contradicting claim as supported. This corrects the
  0.6.0 entry below, which described number guards the code did not contain.
  4 new regression tests (false number rejected, matching number supported,
  Eastern-Arabic digits normalized, negation parity retained); full suite 148.
- assist(): fail-closed salvage of token-truncated model JSON. An unterminated
  answer string is cut back to the last sentence-final punctuation (a tail cut
  mid-number/mid-negation/mid-citation could otherwise verify falsely);
  claims keep only fully closed strings; salvage only ever shrinks content and
  every claim still needs corpus support. A truncation warning is surfaced in
  AssistResult.warnings. Regression tests: mid-sentence cut, fully truncated
  fragment (insufficient), closed-strings-only claims, dangling escape.

# Changelog

## 0.6.3 - 2026-10-26

- collector: meta-tag regex inline flags moved to compile flags; the mid-pattern inline flags are a hard re.error on Python 3.11+ at import. No behavior change. v0.6.2 is superseded before any PyPI publication.

## 0.6.2 - 2026-10-26

- mypy configuration: optional integrations (Pillow, snowballstemmer, sentence-transformers) are guarded optional imports and now carry ignore-missing-imports overrides, since they are not installed in CI. No package behavior change. v0.6.1 is superseded before any PyPI publication.

## 0.6.1 - 2026-10-26

- CI compliance for the 0.6.0 tree: ruff lint and mypy type checks pass again across Python 3.10-3.13 (import sorting, context-managed file handles, explicit subprocess check argument, Optional narrowing, loop-variable disambiguation, ruff target-version pinned to py310). No behavior change; full suite 253 tests.

## 0.6.0 - 2026-10-26

- Add `newsrag.assist`: experimental port of the question-only assistant decision ladder (draft, per-claim lexical support with negation-parity guard, guarded rewrite re-verified against its own extracted claims, answer/extractive/insufficient decision, criteria checklist). `validated_for_release` stays False; lexical support is an automated signal, not entailment.
- 6 new tests with fake transports; full suite 138 tests.

## 0.5.0 - 2026-10-25

- Add `newsrag.gemini.GeminiProvider`: a dependency-free adapter that supplies the `generate` and `embed` callables from Google's Gemini API (`urllib` only, injectable transport, response-shape validation, no key logging). Default model names live-checked against the API on 2026-10-25: `gemini-3.1-flash-lite` + `gemini-embedding-001` (earlier guessed IDs were corrected after live probing); pass explicit names to pin or update them. As with any external provider, only send sources you are allowed to share.
- 6 new tests cover key/name validation, request shape, embedding count and numeric checks, and an end-to-end `NewsroomRAG` run with a stubbed transport.

## 0.4.0 - 2026-10-03

- Add experimental verification lab modules (`newsrag.lab`, `lab_*`, `neural`, `neural_components`) and the optional `neural` extra (NumPy). All outputs are `validated_for_release=False`; no weights or data are shipped.
- Add 17 test files for the lab modules; tighten return types in `core` and `evaluation`.
- README documents measured results only, against the majority baseline.

## 0.3.1 - 2026-09-26

- Update the published package description to match current 0.3 features and limits; no retrieval or API behavior changes from 0.3.0.
- Refresh GitHub release presentation and README wording.

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

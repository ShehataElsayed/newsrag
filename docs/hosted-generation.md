# Bounded generation and source collection (unreleased)

NewsRAG's OpenAI-compatible `QwenProvider` can target a local server or an explicitly selected remote HTTPS service. A server's model string is a report, not independent proof of model identity or answer correctness. No model is downloaded or started by the adapter.

```python
provider = QwenProvider(
    model=chosen_model,
    base_url=chosen_https_endpoint,
    api_key=secret_from_environment,
    max_tokens=256,
    max_prompt_chars=12000,
    timeout=30,
    require_stop=True,
)
rag = NewsroomRAG(generate=provider.generate)
```

This is configuration, not an instruction to enable a provider. Before using any remote service, verify its current free plan, account-wide hard no-charge enforcement, applicable model licence and data policy. Do not send private newsroom material without permission. Never expose the API key to a public browser.

## What the candidate controls

- Fixed maximum prompt characters and requested output tokens, explicit nonstreaming requests, no automatic retry or provider fallback.
- Remote HTTPS only, no endpoint credentials/query strings, no redirect forwarding of bearer credentials, and a 2MB response ceiling.
- Truncated, tool-call, filtered and refusal responses fail closed instead of becoming normal prose. Legacy endpoints omitting `finish_reason` remain supported; that omission is visible as `None`, not proof of a completed answer. Public hosted callers should set `require_stop=True`, which rejects missing completion status. This remains completion-protocol evidence, not semantic truth.
- Per-call `last_status`, `finish_reason` and numeric usage diagnostics. State clears before each attempt, so an earlier success cannot make a later failure look successful. The adapter does not retain prompts or completions.
- Injected transports never set execution/model flags, even when they return believable model metadata.

These bounds are not billing controls. Token counts, neurons and model identity come from the server, can be missing, and must not be interpreted as an exact account-wide quota. Analytics may lag. A local reservation cannot prevent unrelated account users from consuming quota. Public activation must remain disabled until external no-charge enforcement is established.

## Evidence and quality

Protocol tests use synthetic Arabic sources and injected transports. They establish request bounds, fail-closed behavior and preservation of retrieval excerpts, not model accuracy. Reference-valid citations do not establish entailment, source truth or completeness. A successful protocol test does not qualify the full investigation engine.

Autonomous investigation is not validated: collecting a source, generating fluent Arabic and pointing to an existing citation are distinct from proving every claim is supported. Never convert a missing source, failed request or truncated output into an affirmative finding.

## Role-aware local retrieval candidate

`NewsroomRAG` now accepts an optional `query_embed` callable alongside the document `embed` callable. With no `query_embed`, the old behavior is unchanged. When a selected model requires different document/query instructions, configure them explicitly:

```python
adapter = MultilingualEmbedder(
    model=preloaded_reviewed_model,
    document_prefix="passage: ",
    query_prefix="query: ",
)
rag = NewsroomRAG(embed=adapter, query_embed=adapter.embed_query)
```

This prevents a protocol mismatch with models such as multilingual E5. It does not demonstrate that E5 is better on Arabic newsroom tasks. Prefixes affect model inputs only, never stored source text or cited excerpts. Both encoders must use the same reviewed model/vector space. Neither this addition nor a locally built package starts a model download. A caller who uses the lazy adapter without a preloaded model may trigger a download on first nonempty use, which must be an explicit local-mode choice. Official model requirements: https://huggingface.co/intfloat/multilingual-e5-small .

## Collector provenance safety candidate

The declared-source collector now reuses NewsRAG's public-DNS-pinned HTTPS transport instead of unrestricted urllib redirects. It checks robots.txt first, rejects redirects/non-200/unsupported types, caps each response at 2 MB, records empty extraction as failure, and adds SHA256 hashes for raw bytes and extracted UTF8 text. Wikipedia exclusion checks the hostname and subdomains, not arbitrary query text. Publisher metadata remains distinct from speaker attribution; neither is authenticated merely by fetching a page.

Injected openers are explicitly labelled `transport="injected"` in provenance. They remain caller-controlled seams responsible for DNS protections and are not evidence that the production path ran. Tests enforce size/redirect checks for these seams too. No article content is shipped in this candidate.

A robots failure is a visible gap, not permission to impersonate a browser or bypass the publisher. No scheduler/crawler or automatic retry is added. Existing hosts that redirect or reject robots may now fail closed and need a human-reviewed ingestion route.

The web extra explicitly includes `lxml_html_clean` because a fresh Python3.10 environment resolved trafilatura2.3.1/jusText3.0.2/lxml6.1.3 without that separately required import. This fixes the reproduced clean-install failure without changing the package version or licence.

The separate llama-server adapter defaults to loopback endpoints; remote HTTPS requires explicit allow_remote. Loopback alone cannot prove model identity or that the serving process is not a proxy. No local/no-network execution guarantee follows from an HTTP response.

The llama-server transport rejects all redirects (including 301/302/303/307/308), so initial loopback validation cannot be bypassed through an HTTP Location response. Reads are capped at 2 MB. Local HTTP-server regression tests verify that no redirected GET or POST receives the synthetic prompt; these test the wire contract, not model execution.

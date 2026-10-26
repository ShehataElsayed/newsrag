# Semantic claim grading (EXPERIMENTAL)

Status: working investigation tool, not a finished component. Not part of any released NewsRAG version. No model has been demonstrated as good enough to feed it under the resource constraints tested so far.

## Idea
A model (any backend) may only return structured facets with verbatim text spans. This package verifies every span against the source document (NFC + UTF-8 byte offsets, duplicate-span disambiguation) and a fixed, versioned rule set decides bind / deny / abstain. The model never decides directly, and no decision binds without a verified claim span (v1.3).

## Files
- `harness.py` - grading core (span verification, context bundles, anchor extraction, validity gate, decision rules). Current version: v1.3.
- `runner_v2.py` - probe runner: stops at the first infrastructure failure, keeps complete stderr, separates infrastructure failures from model-output failures.
- `selftest.py` - 17 decision-rule contrast tests.
- `selftest-v1.3-grounding.py` - 13 grounding + malformed-input tests.
- `selftest-v1.2-property.py` - property tests (byte-offset round-trips on mixed Arabic/English, normalization, duplicate spans) plus 4000 generated malformed inputs, zero observed crashes in these runs (not a proof for all inputs).
- `RUNTIME-RECIPE-llamacpp-cpu.md` - pinned local CPU runtime recipe (exact model/runtime versions, byte sizes, hashes, licenses, measured resource facts).

## Run the tests
```
python3 selftest.py
python3 selftest-v1.3-grounding.py
python3 selftest-v1.2-property.py
```
All three pass on the code as committed. No third-party dependencies; standard library only.

## Findings so far (honest)
- A 0.5B-parameter local instruct model runs fully offline (~13 s per unit on 2 CPU threads) but its structured output was too unreliable in a 20-unit diagnostic (18 of 20 outputs invalid).
- A 1.5B-parameter model in the tested configuration (default settings, 1860 MiB virtual memory cap, 1982 MB RAM machine) aborted during loading; exact cause unknown. This is a constraint of the tested setup, not evidence about 1.5B models in general.
- Which runtime/engine path is viable is an open decision.

## License
Same as the NewsRAG repository. Third-party models referenced in the runtime recipe are under their own licenses (the referenced Qwen2.5 models are Apache-2.0); no model weights are bundled here.

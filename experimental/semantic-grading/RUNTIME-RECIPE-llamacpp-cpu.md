# Transferable free local AI runtime recipe (CPU, no GPU, no account, no API key)
Status: execution-verified 2026-10-26, Linux x86_64 sandbox, 1982MB RAM, 2 threads. DIAGNOSTIC-grade only; NOT a qualified engine for scoring/quality.

## Pinned components
- Runtime: llama.cpp prebuilt b11897, asset llama-b11897-bin-ubuntu-x64.tar.gz (22.5MB)
  https://github.com/ggml-org/llama.cpp/releases -> llama-b11897-bin-ubuntu-x64.tar.gz
  Ships llama-completion/llama-cli/llama-server + bundled .so. No build (pip llama-cpp-python is source-build-only here; too slow).
- Model: Qwen/Qwen2.5-0.5B-Instruct-GGUF, qwen2.5-0.5b-instruct-q4_k_m.gguf
  https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf
  Exactly 491,400,032 bytes; sha256 starts 74a4da8c9fdbcd15 (verify full hash). Apache-2.0.

## Resource facts (0.5B Q4_K_M, -t 2 -c 2048 -n 220)
- RAM: fits well under 1GB total; safe with 1.7GB free. Re-measure before ctx>2048 or 1.5B.
- Latency: mean 13.3s (9.6-17.3) per call incl. process start, 600-1240 tokens. Small-n diagnostic.
- Network: zero non-loopback connections from llama* during inference (ss -tnp). Offline after download.

## Invocation (raw completion; manual ChatML required - llama-completion applies no chat template)
  export LD_LIBRARY_PATH=<extract_dir>   # mandatory: binary loads bundled .so from its own dir
  PROMPT=<im_start>system\\nSYSTEM<im_end>\\n<im_start>user\\nUSER<im_end>\\n<im_start>assistant\\n  (use the real pipe-bracket tokens)
  <extract_dir>/llama-completion -m model.gguf -c 2048 -n 220 -t 2 --no-display-prompt -p "$PROMPT"

## Gotchas (all observed)
1. stdout ends with a hardcoded trailer line after the model output: blank lines + "> EOF by user". NOT model output; strip at your adapter before parsing.
2. ctx=1024 silently produced EMPTY stdout for long Arabic prompts (Arabic tokenizes heavy). Use >=2048 and treat empty stdout as failure.
3. 0.5B emits JSON-shaped text but semantically sloppy values (multi-role arrays, pipe-joined enums, nested lists). A strict downstream validator must expect ~90% invalid at this size.
4. LD_LIBRARY_PATH must point at the extract dir or the binary fails to start.
5. free -g truncates; check free -m for memory feasibility.

## Transfer notes
No localhost assumptions: both artifacts are plain HTTPS downloads from the URLs above; pin by byte size + sha256. Artifact transfer of the 22.5MB tarball or 491MB GGUF was NOT done (attachment limit 20MB; HF/GitHub re-download at these sizes is seconds-to-minutes on datacenter egress).

## 1.5B fit finding (2026-10-26)
q4_k_m 1.5B (1,117,320,736 B, sha256 6a1a2eb6...9407e, Apache-2.0) SIGABRTs ~2s into load on a 1982 MiB total / ~1700 MiB available box, tested at default ctx2048 buffers under RLIMIT_AS 1860 MiB; exact cause unknown (stderr cause not retained). Cumulative child max RSS ~1.5-1.75GB was observed but cannot be attributed per unit. Result: tested configuration failed under current resource constraints - NOT evidence of universal 1.5B infeasibility. Smaller buffers or q3_k_m (881.6 MiB) are untested options, not established fits. 0.5B remains the only demonstrated local GGUF in this environment.

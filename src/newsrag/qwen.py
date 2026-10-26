"""Qwen2.5 generation adapter (OpenAI-compatible chat completions).

Targets a local Ollama server (default, no key, no data leaves the machine) or
any OpenAI-compatible endpoint the user explicitly configures. Remote hosts
must use HTTPS - confidential newsroom inputs must never go to an external
provider in cleartext or without an explicit user choice.

EXECUTION STATUS: this adapter is configuration + wire protocol only. No model
weights are bundled, downloaded, or claimed to have run. ``response_received``
means only that the configured endpoint returned a parseable chat-completion -
a mock, a proxy, or a different model could produce that, so it NEVER proves
the exact named model executed. ``model_reported`` merely echoes the model id
the server claims. Independent confirmation of exact-model execution requires
evidence this adapter cannot supply (e.g. operator verification of the serving
process); both flags stay False in tests and under injected transports.

Local-execution caveat: the localhost default keeps the GENERATION call on the
machine, but says nothing about retrieval - if the index/search layer reaches
remote services, those inputs are not local. Locality is per-component, not
pipeline-wide. Model identity/license (e.g. Qwen2.5-1.5B-Instruct, Apache-2.0)
must be verified on the official model card before use; availability of an
endpoint is not deployment permission.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any
from urllib import request

from .core import ValidationError

_LOCAL_HOSTS = ("localhost", "127.0.0.1", "::1")


class QwenProvider:
    """``generate`` adapter for :class:`NewsroomRAG` over chat completions.

    ``transport`` is injectable for tests: it receives a
    ``urllib.request.Request`` and returns the decoded JSON body.
    """

    def __init__(self, *, model: str = "qwen2.5:1.5b-instruct",
                 base_url: str = "http://localhost:11434/v1",
                 api_key: str | None = None, timeout: float = 60.0,
                 transport: Callable[[request.Request], Any] | None = None):
        if not isinstance(model, str) or not model.strip():
            raise ValidationError("QwenProvider needs a non-empty model name")
        if not isinstance(base_url, str) or "://" not in base_url:
            raise ValidationError("QwenProvider needs a base_url with a scheme")
        scheme, host = base_url.split("://", 1)
        host = host.split("/", 1)[0].split(":", 1)[0]
        if scheme != "https" and host not in _LOCAL_HOSTS:
            raise ValidationError(
                "remote Qwen endpoints require HTTPS; plain HTTP is allowed only for localhost")
        if host not in _LOCAL_HOSTS and not (api_key and api_key.strip()):
            raise ValidationError(
                "remote Qwen endpoints need an explicit API key; do not send inputs without one")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._key = api_key
        self.timeout = timeout
        self._transport = transport or self._post
        self._injected_transport = transport is not None
        # Truthful execution bookkeeping: a parsed response never proves the exact
        # named model ran (mock/proxy/other model). Injected (test) transports
        # never set these flags.
        self.response_received = False
        self.model_reported: str | None = None

    def _post(self, req: request.Request) -> Any:
        with request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def generate(self, prompt: str) -> str:
        """One chat-completion call. Fail-closed: malformed responses raise."""
        headers = {"content-type": "application/json"}
        if self._key:
            headers["authorization"] = f"Bearer {self._key}"
        req = request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps({
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2,
            }).encode("utf-8"),
            headers=headers, method="POST")
        body = self._transport(req)
        try:
            content = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValidationError(f"Qwen endpoint returned an unusable response: {exc}") from exc
        if not isinstance(content, str) or not content.strip():
            raise ValidationError("Qwen endpoint returned an empty completion")
        if not self._injected_transport:
            self.response_received = True
            self.model_reported = body.get("model") if isinstance(body, dict) else None
        return content


def llamacpp_generate(model_path: str, *, binary: str = "llama-completion",
                      n_ctx: int = 1024, n_predict: int = 256, threads: int = 2,
                      timeout: float = 180.0, temperature: float = 0.0,
                      system: str | None = None,
                      json_schema: dict | None = None,
                      grammar_file: str | None = None,
                      extra_flags: list[str] | None = None) -> Callable[[str], str]:
    """Local generation via a llama.cpp CLI binary over a local GGUF file.

    REAL local execution: the named GGUF file is actually loaded and run on
    this machine - no network, no key, inputs never leave the process. This is
    distinct from :class:`QwenProvider`, which only talks to an endpoint and
    cannot prove which model answered. Fail-closed: missing binary/model or an
    empty/unparseable completion raises ValidationError.
    """
    import subprocess
    from pathlib import Path

    if not Path(model_path).is_file():
        raise ValidationError(f"GGUF model not found: {model_path}")

    def _generate(prompt: str) -> str:
        cmd = [binary, "-m", model_path, "-c", str(n_ctx), "-t", str(threads),
               "-n", str(n_predict), "--temp", str(temperature), "-p", prompt]
        if system:
            cmd += ["-sys", system]
        if json_schema is not None:
            # constrained generation: the runtime enforces the schema at decode
            # time, so output is structurally valid JSON by construction - this
            # is not parse-tolerance and never merges/bypasses content guards.
            cmd += ["--json-schema", json.dumps(json_schema, ensure_ascii=False)]
        if grammar_file is not None:
            # GBNF grammar file: same constrained-decoding guarantee, for llama.cpp
            # builds whose --json-schema conversion path is broken (e.g. b11889).
            cmd += ["--grammar-file", grammar_file]
        if extra_flags:
            cmd += list(extra_flags)
        try:
            # stdin=DEVNULL: llama-completion re-enters an interactive prompt
            # after answering when stdin is an open TTY, which hangs callers
            # running inside interactive terminals.
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout,
                stdin=subprocess.DEVNULL, check=False)
        except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
            raise ValidationError(f"llama.cpp run failed: {exc}") from exc
        out = proc.stdout or ""
        # conversation mode echoes "user\n<prompt>\nassistant\n<reply>"; take the last assistant block
        marker = "assistant\n"
        if marker in out:
            reply = out.rsplit(marker, 1)[1]
            reply = reply.split("\n> EOF")[0].split("\nuser\n")[0].strip()
        else:
            # newer llama-cli echoes "> <prompt>" verbatim before the reply (and
            # prints a startup banner); cut everything through the echoed prompt
            # so prompt text (which may contain braces) is never parsed as output.
            # the echo may be TRUNCATED (llama-cli elides long prompts with
            # " ... (truncated)"), so head anchors can vanish: first try tail
            # anchors - the prompt's final line (the actual question/text) is
            # the last thing echoed before the reply.
            reply = ""
            for k in (min(140, len(prompt)), 80, 40):
                if k <= 0:
                    continue
                anchor = prompt[-k:]
                idx = out.rfind(anchor)
                if idx != -1:
                    reply = out[idx + len(anchor):].strip()
                    break
            for k in (len(prompt), 240, 160, 100, 60):
                if reply:
                    break
                anchor = prompt[:k]
                idx = out.rfind(anchor)
                if idx != -1:
                    reply = out[idx + len(anchor):]
                    # the echo may continue past the anchor (wrapped); drop up to
                    # the first newline that precedes a JSON opener or text start
                    if k < len(prompt):
                        nl = reply.find("\n")
                        if nl != -1:
                            reply = reply[nl + 1:]
                    reply = reply.strip()
                    break
            if not reply:
                reply = out.strip()
        if proc.returncode != 0 or not reply:
            raise ValidationError(f"llama.cpp returned no usable completion (rc={proc.returncode})")
        return reply

    return _generate


def llamacpp_server_generate(*, base_url: str = "http://127.0.0.1:18099",
                             n_predict: int = 220, temperature: float = 0.0,
                             grammar_file: str | None = None,
                             timeout: float = 120.0) -> Callable[[str], str]:
    """Local generation against a PERSISTENT llama-server on localhost.

    Same local-execution guarantee as llamacpp_generate (the server loads the
    named GGUF on this machine; nothing leaves the process), but the model is
    loaded once instead of per call: on small hardware the per-call reload
    dominates (15s -> ~2s per extraction, diagnosis). Grammar is passed
    per request, so one server serves every constrained task. The caller owns
    the server lifecycle (start/stop around the run)."""
    from pathlib import Path
    grammar = None
    if grammar_file is not None:
        grammar = Path(grammar_file).read_text(encoding="utf-8")

    def _generate(prompt: str) -> str:
        body: dict[str, Any] = {"prompt": prompt, "n_predict": n_predict,
                                "temperature": temperature, "cache_prompt": True}
        if grammar is not None:
            body["grammar"] = grammar
        req = request.Request(
            base_url.rstrip("/") + "/completion",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST")
        with request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        content = payload.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValidationError("llama-server returned an empty completion")
        return content

    return _generate

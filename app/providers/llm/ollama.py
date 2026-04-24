from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request

from app.providers.llm.base import LLMProvider, LLMRequest

log = logging.getLogger(__name__)


class OllamaProvider(LLMProvider):
    """LLMProvider backed by a local Ollama server.

    Ollama exposes a POST /api/chat endpoint that accepts OpenAI-style
    messages. We talk to it with stdlib urllib to avoid adding a new
    dependency for a single HTTP call.

    Model strings follow Ollama's pull/run format, including
    Hugging Face hosted models such as
    ``hf.co/mradermacher/Llama-Poro-2-8B-Instruct-GGUF:Q6_K``.
    """

    def __init__(self, url: str, model: str, timeout: float = 180.0) -> None:
        if not url:
            raise ValueError("OLLAMA_URL is required for OllamaProvider")
        if not model:
            raise ValueError("OLLAMA_MODEL is required for OllamaProvider")
        self._url = url.rstrip("/")
        self._model = model
        self._timeout = timeout

    def complete(self, request: LLMRequest) -> str:
        messages: list[dict[str, str]] = [{"role": "system", "content": request.system}]
        messages.extend(
            {"role": turn.role, "content": turn.content} for turn in request.history
        )
        messages.append({"role": "user", "content": request.user_message})

        body = {
            "model": self._model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": request.temperature,
                "num_predict": request.max_tokens,
            },
        }

        log.debug(
            "ollama.chat model=%s url=%s history_turns=%d",
            self._model, self._url, len(request.history),
        )
        req = urllib.request.Request(
            f"{self._url}/api/chat",
            data=json.dumps(body).encode("utf-8"),
            headers={"content-type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Ollama HTTP {exc.code} at {self._url}/api/chat: {detail}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Ollama request failed ({self._url}/api/chat): {exc.reason}"
            ) from exc

        message = payload.get("message") or {}
        content = message.get("content") or ""
        return content.strip()

from __future__ import annotations

import io
import json
from typing import Any

import pytest

from app.domain import ChatTurn
from app.providers.llm import LLMRequest, OllamaProvider


class _FakeResponse:
    def __init__(self, body: dict[str, Any]):
        self._body = json.dumps(body).encode("utf-8")

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *exc: object) -> None:
        return None


def test_ollama_builds_request_body_and_parses_response(monkeypatch):
    captured: dict[str, Any] = {}

    def fake_urlopen(req, timeout):  # noqa: ARG001
        captured["url"] = req.full_url
        captured["method"] = req.get_method()
        captured["headers"] = {k.lower(): v for k, v in req.header_items()}
        captured["body"] = json.loads(req.data.decode("utf-8"))
        return _FakeResponse(
            {"model": "poro-test", "message": {"role": "assistant", "content": "  Hei! "}}
        )

    monkeypatch.setattr("app.providers.llm.ollama.urllib.request.urlopen", fake_urlopen)

    provider = OllamaProvider(url="http://localhost:11434/", model="poro-test")
    reply = provider.complete(
        LLMRequest(
            system="sys-prompt",
            history=[ChatTurn(role="user", content="aiempi")],
            user_message="uusi kysymys",
            max_tokens=64,
            temperature=0.3,
        )
    )

    assert reply == "Hei!"
    assert captured["url"] == "http://localhost:11434/api/chat"
    assert captured["method"] == "POST"
    assert captured["headers"]["content-type"] == "application/json"

    body = captured["body"]
    assert body["model"] == "poro-test"
    assert body["stream"] is False
    assert body["options"] == {"temperature": 0.3, "num_predict": 64}
    assert body["messages"] == [
        {"role": "system", "content": "sys-prompt"},
        {"role": "user", "content": "aiempi"},
        {"role": "user", "content": "uusi kysymys"},
    ]


def test_ollama_clamps_num_predict_to_local_max(monkeypatch):
    captured: dict[str, Any] = {}

    def fake_urlopen(req, timeout):  # noqa: ARG001
        captured["body"] = json.loads(req.data.decode("utf-8"))
        return _FakeResponse({"message": {"content": "ok"}})

    monkeypatch.setattr("app.providers.llm.ollama.urllib.request.urlopen", fake_urlopen)

    provider = OllamaProvider(
        url="http://localhost:11434",
        model="poro-test",
        max_tokens=128,
    )
    provider.complete(LLMRequest(system="s", history=[], user_message="q"))

    assert captured["body"]["options"]["num_predict"] == 128


def test_ollama_requires_url_and_model():
    with pytest.raises(ValueError):
        OllamaProvider(url="", model="x")
    with pytest.raises(ValueError):
        OllamaProvider(url="http://x", model="")


def test_ollama_wraps_url_errors_in_runtime_error(monkeypatch):
    import urllib.error

    def boom(req, timeout):  # noqa: ARG001
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr("app.providers.llm.ollama.urllib.request.urlopen", boom)

    provider = OllamaProvider(url="http://localhost:11434", model="poro-test")
    with pytest.raises(RuntimeError, match="Ollama request failed"):
        provider.complete(
            LLMRequest(system="s", history=[], user_message="q")
        )


def test_ollama_wraps_timeout_in_runtime_error(monkeypatch):
    def boom(req, timeout):  # noqa: ARG001
        raise TimeoutError("timed out")

    monkeypatch.setattr("app.providers.llm.ollama.urllib.request.urlopen", boom)

    provider = OllamaProvider(
        url="http://localhost:11434",
        model="poro-test",
        timeout=1,
    )
    with pytest.raises(RuntimeError, match="Ollama request timed out after 1s"):
        provider.complete(LLMRequest(system="s", history=[], user_message="q"))


def test_ollama_handles_http_error(monkeypatch):
    import urllib.error

    def boom(req, timeout):  # noqa: ARG001
        raise urllib.error.HTTPError(
            url=req.full_url, code=404, msg="Not Found", hdrs=None,
            fp=io.BytesIO(b'{"error":"model not pulled"}'),
        )

    monkeypatch.setattr("app.providers.llm.ollama.urllib.request.urlopen", boom)

    provider = OllamaProvider(url="http://localhost:11434", model="missing-model")
    with pytest.raises(RuntimeError, match="Ollama HTTP 404"):
        provider.complete(LLMRequest(system="s", history=[], user_message="q"))

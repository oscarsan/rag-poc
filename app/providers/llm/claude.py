from __future__ import annotations

import logging

from anthropic import Anthropic

from app.providers.llm.base import LLMProvider, LLMRequest

log = logging.getLogger(__name__)


class ClaudeProvider(LLMProvider):
    def __init__(self, api_key: str, model: str) -> None:
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY is required for ClaudeProvider")
        self._client = Anthropic(api_key=api_key)
        self._model = model

    def complete(self, request: LLMRequest) -> str:
        messages = [
            {"role": turn.role, "content": turn.content} for turn in request.history
        ]
        messages.append({"role": "user", "content": request.user_message})

        log.debug(
            "claude.complete model=%s history_turns=%d", self._model, len(request.history)
        )
        response = self._client.messages.create(
            model=self._model,
            system=request.system,
            messages=messages,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
        )
        parts = [block.text for block in response.content if getattr(block, "type", None) == "text"]
        return "".join(parts).strip()

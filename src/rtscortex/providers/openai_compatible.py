"""OpenAI-compatible structured-output provider."""

from __future__ import annotations

import os
from typing import Any

import httpx
from pydantic import BaseModel, Field

from rtscortex.contracts.interfaces import ResponseT


class _Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class _Message(BaseModel):
    content: str


class _Choice(BaseModel):
    message: _Message


class _ChatCompletion(BaseModel):
    choices: list[_Choice] = Field(min_length=1)
    usage: _Usage | None = None


class OpenAICompatibleProvider:
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key_env: str,
        timeout_seconds: float,
        max_tokens: int | None = None,
        enable_thinking: bool | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key_env = api_key_env
        self.max_tokens = max_tokens
        self.enable_thinking = enable_thinking
        self.client = httpx.AsyncClient(timeout=timeout_seconds, transport=transport)
        self.last_usage: dict[str, int] | None = None

    async def generate(
        self,
        response_type: type[ResponseT],
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> ResponseT:
        self.last_usage = None
        api_key = os.environ.get(self.api_key_env, "")
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        request_payload: dict[str, Any] = {
            "model": self.model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": response_type.__name__,
                    "strict": True,
                    "schema": response_type.model_json_schema(),
                },
            },
        }
        if self.max_tokens is not None:
            request_payload["max_tokens"] = self.max_tokens
        if self.enable_thinking is not None:
            request_payload["chat_template_kwargs"] = {"enable_thinking": self.enable_thinking}
        response = await self.client.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=request_payload,
        )
        response.raise_for_status()
        completion = _ChatCompletion.model_validate(response.json())
        if completion.usage is not None:
            self.last_usage = completion.usage.model_dump()
        return response_type.model_validate_json(completion.choices[0].message.content)

    async def close(self) -> None:
        await self.client.aclose()

from __future__ import annotations

import asyncio
import json

import httpx
import pytest
from pydantic import ValidationError

from rtscortex.agents import PlanningOutput
from rtscortex.providers import OpenAICompatibleProvider


def make_provider(handler: httpx.AsyncBaseTransport) -> OpenAICompatibleProvider:
    return OpenAICompatibleProvider(
        base_url="http://model.test/v1",
        model="test-model",
        api_key_env="RTSCORTEX_TEST_API_KEY",
        timeout_seconds=0.1,
        transport=handler,
    )


def test_openai_provider_parses_structured_output_and_usage() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        content = json.dumps(
            {
                "strategic_goal": "Hold the ramp",
                "steps": ["Defend"],
                "proposed_actions": [],
            }
        )
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": content}}],
                "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
            },
        )

    async def execute() -> None:
        provider = make_provider(httpx.MockTransport(handler))
        try:
            result = await provider.generate(
                PlanningOutput,
                system_prompt="plan",
                user_prompt="{}",
            )
            assert result.strategic_goal == "Hold the ramp"
            assert provider.last_usage == {
                "prompt_tokens": 11,
                "completion_tokens": 7,
                "total_tokens": 18,
            }
        finally:
            await provider.close()

    asyncio.run(execute())


@pytest.mark.parametrize("status_code", [429, 500])
def test_openai_provider_surfaces_http_errors(status_code: int) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code, request=request)

    async def execute() -> None:
        provider = make_provider(httpx.MockTransport(handler))
        try:
            with pytest.raises(httpx.HTTPStatusError):
                await provider.generate(PlanningOutput, system_prompt="plan", user_prompt="{}")
        finally:
            await provider.close()

    asyncio.run(execute())


def test_openai_provider_surfaces_timeout() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("model timed out", request=request)

    async def execute() -> None:
        provider = make_provider(httpx.MockTransport(handler))
        try:
            with pytest.raises(httpx.ReadTimeout):
                await provider.generate(PlanningOutput, system_prompt="plan", user_prompt="{}")
        finally:
            await provider.close()

    asyncio.run(execute())


def test_openai_provider_rejects_invalid_structured_output() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            request=request,
            json={"choices": [{"message": {"content": "{}"}}]},
        )

    async def execute() -> None:
        provider = make_provider(httpx.MockTransport(handler))
        try:
            with pytest.raises(ValidationError):
                await provider.generate(PlanningOutput, system_prompt="plan", user_prompt="{}")
        finally:
            await provider.close()

    asyncio.run(execute())

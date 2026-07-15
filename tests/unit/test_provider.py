from __future__ import annotations

import asyncio
import json

import httpx
import pytest
from pydantic import ValidationError

from rtscortex.agents import PlanningOutput
from rtscortex.agents.models import planning_output_model
from rtscortex.contracts import ActionArgumentType, AvailableAction
from rtscortex.providers import FakeProvider, OpenAICompatibleProvider
from tests.helpers import make_observation


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


def test_openai_provider_sends_configured_qwen_generation_options() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["max_tokens"] == 256
        assert payload["chat_template_kwargs"] == {"enable_thinking": False}
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "strategic_goal": "Hold",
                                    "steps": [],
                                    "proposed_actions": [],
                                }
                            )
                        }
                    }
                ]
            },
        )

    async def execute() -> None:
        provider = OpenAICompatibleProvider(
            base_url="http://model.test/v1",
            model="Qwen/Qwen3-8B",
            api_key_env="RTSCORTEX_TEST_API_KEY",
            timeout_seconds=0.1,
            max_tokens=256,
            enable_thinking=False,
            transport=httpx.MockTransport(handler),
        )
        try:
            await provider.generate(PlanningOutput, system_prompt="plan", user_prompt="{}")
        finally:
            await provider.close()

    asyncio.run(execute())


def test_openai_provider_sends_action_bound_planning_schema() -> None:
    base = make_observation()
    observation = base.model_copy(
        update={
            "state": base.state.model_copy(
                update={
                    "economy": base.state.economy.model_copy(
                        update={"supply_used": 11, "supply_cap": 15}
                    )
                }
            ),
            "available_actions": [
                AvailableAction(
                    name="Build_Pylon_Screen",
                    argument_names=["screen"],
                    argument_types=[ActionArgumentType.POSITION],
                    actor_scopes=["Builder/Builder-Probe-1"],
                    argument_candidates=[[[60, 40]]],
                ),
                AvailableAction(
                    name="Train_Zealot",
                    actor_scopes=["Developer/Empty"],
                ),
            ],
        }
    )
    response_type = planning_output_model(observation)

    async def handler(request: httpx.Request) -> httpx.Response:
        schema = json.loads(request.content)["response_format"]["json_schema"]["schema"]
        proposal_refs = schema["properties"]["proposed_actions"]["items"]["anyOf"]
        proposal_schemas = [
            schema["$defs"][reference["$ref"].rsplit("/", maxsplit=1)[-1]]
            for reference in proposal_refs
        ]
        proposals_by_name = {
            proposal["properties"]["name"]["const"]: proposal for proposal in proposal_schemas
        }
        build_schema = proposals_by_name["Build_Pylon_Screen"]
        assert build_schema["properties"]["actor"]["const"] == ("Builder/Builder-Probe-1")
        build_arguments = build_schema["properties"]["arguments"]
        assert build_arguments["minItems"] == build_arguments["maxItems"] == 1
        position_schema = build_arguments["prefixItems"][0]
        assert position_schema["minItems"] == position_schema["maxItems"] == 2
        assert [item["const"] for item in position_schema["prefixItems"]] == [60, 40]
        train_schema = proposals_by_name["Train_Zealot"]
        assert train_schema["properties"]["actor"]["const"] == "Developer/Empty"
        train_arguments = train_schema["properties"]["arguments"]
        assert train_arguments["minItems"] == train_arguments["maxItems"] == 0
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "strategic_goal": "Build supply",
                                    "steps": ["Build one Pylon"],
                                    "proposed_actions": [
                                        {
                                            "actor": "Builder/Builder-Probe-1",
                                            "name": "Build_Pylon_Screen",
                                            "arguments": [[60, 40]],
                                        }
                                    ],
                                }
                            )
                        }
                    }
                ]
            },
        )

    async def execute() -> None:
        provider = make_provider(httpx.MockTransport(handler))
        try:
            output = await provider.generate(
                response_type,
                system_prompt="plan",
                user_prompt="{}",
            )
            assert output.proposed_actions[0].name == "Build_Pylon_Screen"
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


def test_fake_provider_uses_available_live_actor_scope() -> None:
    async def execute() -> None:
        provider = FakeProvider()
        result = await provider.generate(
            PlanningOutput,
            system_prompt="plan",
            user_prompt=json.dumps(
                {
                    "observation": {
                        "available_actions": [
                            {
                                "name": "Attack_Unit",
                                "actor_scopes": [
                                    "CombatGroupSmac/Stalker-1",
                                    "CombatGroupSmac/Zealot-1",
                                ],
                            }
                        ],
                        "state": {"visible_enemies": [{"unit_id": "0xabc"}]},
                    }
                }
            ),
        )
        assert result.proposed_actions[0].actor == "CombatGroupSmac/Stalker-1"

    asyncio.run(execute())

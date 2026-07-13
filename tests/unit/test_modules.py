from __future__ import annotations

import asyncio
from typing import NoReturn

from rtscortex.agents import ReflectionModule
from rtscortex.contracts.interfaces import AgentContext, ResponseT
from tests.helpers import make_observation


class FailingProvider:
    async def generate(
        self,
        response_type: type[ResponseT],
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> NoReturn:
        del response_type, system_prompt, user_prompt
        raise AssertionError("provider should not be called on the first step")


def test_reflection_skips_first_decision() -> None:
    module = ReflectionModule(FailingProvider())
    result = asyncio.run(module.run(AgentContext(observation=make_observation())))
    assert result.updates == {"reflection": None, "lessons": []}

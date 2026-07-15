"""Upstream LLMAgent query override that delegates decisions to RTSCortex."""

from __future__ import annotations

from typing import Any, Protocol, cast


class RuntimeDecisionBroker(Protocol):
    def submit(self, agent: Any, obs: Any, text_observation: str) -> str: ...


class RuntimeQueryMixin:
    """Replace only the upstream model-query phase.

    Observation translation and action translation stay upstream-owned. The
    shared broker is the only component allowed to call the runtime API.
    """

    broker: RuntimeDecisionBroker
    lock: Any
    is_waiting: bool
    first_action: bool
    action_lists: list[Any]

    def query(self, obs: Any) -> None:
        agent = cast(_UpstreamQueryAgent, self)
        with agent.lock:
            agent.is_waiting = True
        try:
            agent.get_text_c_inp()
            text_observation = agent.get_text_o(obs)
            action_text = agent.broker.submit(agent, obs, text_observation)
            action_lists, _ = agent.get_func_a(action_text)
            agent.first_action = True
            with agent.lock:
                agent.action_lists = action_lists
        finally:
            with agent.lock:
                agent.is_waiting = False


class _UpstreamQueryAgent(Protocol):
    broker: RuntimeDecisionBroker
    lock: Any
    is_waiting: bool
    first_action: bool
    action_lists: list[Any]

    def get_text_c_inp(self) -> None: ...

    def get_text_o(self, obs: Any) -> str: ...

    def get_func_a(self, raw_text_a: str) -> tuple[list[Any], dict[str, Any]]: ...

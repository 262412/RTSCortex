"""Agent runtime orchestration."""

from typing import TYPE_CHECKING, Any

from rtscortex.runtime.engine import RuntimeEngine

if TYPE_CHECKING:
    from rtscortex.runtime.cortex_engine import CortexRuntimeEngine

__all__ = ["CortexRuntimeEngine", "RuntimeEngine"]


def __getattr__(name: str) -> Any:
    if name == "CortexRuntimeEngine":
        from rtscortex.runtime.cortex_engine import CortexRuntimeEngine

        return CortexRuntimeEngine
    raise AttributeError(name)

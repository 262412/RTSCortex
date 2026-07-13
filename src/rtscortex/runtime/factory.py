"""Runtime construction from experiment configuration."""

from pathlib import Path

from rtscortex.config import ExperimentConfig
from rtscortex.contracts import LLMProvider
from rtscortex.memory import EventStore
from rtscortex.providers import FakeProvider, OpenAICompatibleProvider
from rtscortex.runtime.engine import RuntimeEngine


def build_runtime(config: ExperimentConfig, run_dir: Path) -> RuntimeEngine:
    store = EventStore(run_dir / "events.sqlite3", run_dir / "events.jsonl")
    provider: LLMProvider
    if config.provider.kind == "fake":
        provider = FakeProvider()
    else:
        provider = OpenAICompatibleProvider(
            base_url=config.provider.base_url,
            model=config.provider.model,
            api_key_env=config.provider.api_key_env,
            timeout_seconds=config.provider.timeout_seconds,
            max_tokens=config.provider.max_tokens,
            enable_thinking=config.provider.enable_thinking,
        )
    return RuntimeEngine(config=config, store=store, provider=provider)

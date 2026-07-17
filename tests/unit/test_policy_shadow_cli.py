from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from typer.testing import CliRunner

from rtscortex.agents.models import PlanningOutput
from rtscortex.cli import app as cli_module
from rtscortex.contracts.interfaces import ResponseT
from tests.helpers import make_observation


def _write_observation_journal(run_dir: Path, *, count: int = 3) -> None:
    run_dir.mkdir()
    observation = make_observation()
    entries = []
    for index in range(count):
        current = observation.model_copy(update={"step_id": index, "game_loop": index * 16})
        entries.append(
            {
                "event_id": index + 1,
                "run_id": current.run_id,
                "episode_id": current.episode_id,
                "step_id": current.step_id,
                "event_type": "observation",
                "created_at": f"2026-01-01T00:00:0{index}+00:00",
                "payload": current.model_dump(mode="json"),
            }
        )
    (run_dir / "events.jsonl").write_text(
        "".join(json.dumps(entry) + "\n" for entry in entries),
        encoding="utf-8",
    )


def test_policy_shadow_cli_writes_one_offline_no_download_artifact(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "historical-run"
    _write_observation_journal(run_dir)
    output = tmp_path / "comparison.json"

    result = CliRunner().invoke(
        cli_module.app,
        [
            "policy-shadow",
            str(run_dir),
            "--no-current-qwen",
            "--limit",
            "1",
            "--stride",
            "2",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    assert f"Artifact: {output}" in result.output
    comparison = json.loads(output.read_text(encoding="utf-8"))
    assert comparison["candidate_ids"] == [
        "qwen3-8b-current",
        "hima-protoss-a",
        "hima-protoss-b",
        "hima-protoss-c",
        "hiernet-sc2-protoss",
    ]
    assert len(comparison["fixtures"]) == 1
    assert comparison["fixtures"][0]["goal_spec"]["goal_id"] == "protoss-opening-v1"
    assert [
        requirement["action_name"]
        for requirement in comparison["fixtures"][0]["goal_spec"]["requirements"]
    ] == ["Build_Pylon_Screen", "Build_Gateway_Screen", "Train_Zealot"]
    statuses = [record["status"] for record in comparison["records"]]
    assert statuses == ["skipped", "unavailable", "unavailable", "unavailable", "unavailable"]
    assert all(
        "no download attempted" in record["availability"]["reason"]
        for record in comparison["records"][1:]
    )


class RecordingOpenAIProvider:
    instances: list[RecordingOpenAIProvider] = []

    def __init__(self, **settings: Any) -> None:
        self.settings = settings
        self.closed = False
        self.user_payloads: list[dict[str, Any]] = []
        self.instances.append(self)

    async def generate(
        self,
        response_type: type[ResponseT],
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> ResponseT:
        assert "goal_progress" in system_prompt
        self.user_payloads.append(json.loads(user_prompt))
        return response_type.model_validate(
            PlanningOutput(
                strategic_goal="Follow deterministic opening progress",
                proposed_actions=[],
            ).model_dump()
        )

    async def close(self) -> None:
        self.closed = True


def test_policy_shadow_cli_builds_and_closes_qwen_from_run_config(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    run_dir = tmp_path / "qwen-run"
    _write_observation_journal(run_dir, count=1)
    (run_dir / "config.yaml").write_text(
        """
provider:
  kind: openai_compatible
  model: local-qwen3-8b
  base_url: http://127.0.0.1:8000/v1
  api_key_env: TEST_QWEN_KEY
  timeout_seconds: 12
  max_tokens: 300
  enable_thinking: false
""",
        encoding="utf-8",
    )
    RecordingOpenAIProvider.instances.clear()
    monkeypatch.setattr(cli_module, "OpenAICompatibleProvider", RecordingOpenAIProvider)

    result = CliRunner().invoke(
        cli_module.app,
        ["policy-shadow", str(run_dir), "--limit", "1"],
    )

    assert result.exit_code == 0, result.output
    provider = RecordingOpenAIProvider.instances[0]
    assert provider.settings == {
        "base_url": "http://127.0.0.1:8000/v1",
        "model": "local-qwen3-8b",
        "api_key_env": "TEST_QWEN_KEY",
        "timeout_seconds": 12.0,
        "max_tokens": 300,
        "enable_thinking": False,
    }
    assert provider.closed is True
    assert provider.user_payloads[0]["goal_spec"]["goal_id"] == "protoss-opening-v1"
    assert provider.user_payloads[0]["goal_progress"]["goal_id"] == "protoss-opening-v1"
    artifact = json.loads((run_dir / "policy-shadow-comparison.json").read_text(encoding="utf-8"))
    assert artifact["records"][0]["status"] == "completed"
    assert [record["status"] for record in artifact["records"][1:]] == [
        "unavailable",
        "unavailable",
        "unavailable",
        "unavailable",
    ]

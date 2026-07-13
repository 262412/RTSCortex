from __future__ import annotations

from typing import Any

from rtscortex_llm_pysc2.protocol import render_action_batch


def test_bridge_renders_llm_pysc2_action_text() -> None:
    batch: dict[str, Any] = {
        "commands": [
            {"actor": "CombatGroup1", "name": "Attack_Unit", "arguments": ["0x123"]},
            {"actor": "CombatGroup1", "name": "Move_Screen", "arguments": [[2, 3]]},
        ]
    }
    assert render_action_batch(batch) == (
        "Actions:\n"
        "    Team CombatGroup1:\n"
        "        <Attack_Unit(0x123)>\n"
        "        <Move_Screen([2,3])>"
    )

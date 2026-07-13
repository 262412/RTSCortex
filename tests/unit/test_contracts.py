from __future__ import annotations

import pytest
from pydantic import ValidationError

from rtscortex.contracts import ObservationEnvelope, UnitState
from tests.helpers import make_observation


def test_observation_round_trip_is_lossless() -> None:
    observation = make_observation(alerts=["under_attack"])
    restored = ObservationEnvelope.model_validate_json(observation.model_dump_json())
    assert restored == observation


def test_contract_rejects_unknown_protocol_version() -> None:
    payload = make_observation().model_dump(mode="json")
    payload["protocol_version"] = "2.0"
    with pytest.raises(ValidationError):
        ObservationEnvelope.model_validate(payload)


def test_unit_health_is_bounded() -> None:
    with pytest.raises(ValidationError):
        UnitState(
            unit_id="bad",
            unit_type="Adept",
            alliance="self",
            health_fraction=1.1,
        )

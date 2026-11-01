"""Contract tests for the Pydantic v2 state schema."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from maritime_sentinel.config import DEFAULT_SCENARIO, scenario_dict
from maritime_sentinel.schemas import (
    LaneSpec,
    NetworkSpec,
    ScenarioSpec,
    StressSpec,
    WeatherZoneSpec,
)


def test_default_scenario_round_trips_without_loss(scenario: ScenarioSpec) -> None:
    payload = json.loads(scenario.model_dump_json())
    rebuilt = ScenarioSpec.model_validate(payload)
    assert rebuilt == scenario
    assert rebuilt.nominal_daily_teu() == pytest.approx(18900.0, rel=1e-6)


def test_scenario_dict_is_json_safe() -> None:
    payload = scenario_dict()
    assert json.loads(json.dumps(payload))["spec_id"] == DEFAULT_SCENARIO.spec_id


def test_unknown_fields_are_rejected(scenario: ScenarioSpec) -> None:
    payload = scenario.model_dump(mode="json")
    payload["stress"]["unexpected_knob"] = 1.0
    with pytest.raises(ValidationError):
        ScenarioSpec.model_validate(payload)


def test_transition_matrix_must_be_stochastic(scenario: ScenarioSpec) -> None:
    payload = scenario.model_dump(mode="json")
    payload["network"]["zones"][0]["transition"][0] = [0.5, 0.5, 0.5, 0.5]
    with pytest.raises(ValidationError, match="sums to"):
        ScenarioSpec.model_validate(payload)


def test_operational_factors_must_be_monotone() -> None:
    with pytest.raises(ValidationError, match="non-increasing"):
        WeatherZoneSpec(
            id="X",
            label="X",
            mean_dwell_hours=[1, 1, 1, 1],
            transition=[[0, 1, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]],
            speed_factor=[1.0, 0.9, 0.95, 0.3],
            productivity_factor=[1.0, 0.9, 0.8, 0.7],
        )


def test_lane_endpoints_must_exist(scenario: ScenarioSpec) -> None:
    payload = scenario.model_dump(mode="json")
    payload["network"]["lanes"][0]["a"] = "ATLANTIS"
    with pytest.raises(ValidationError, match="unknown node"):
        ScenarioSpec.model_validate(payload)


def test_self_loop_lane_rejected(scenario: ScenarioSpec) -> None:
    payload = scenario.model_dump(mode="json")
    payload["network"]["lanes"][0]["b"] = payload["network"]["lanes"][0]["a"]
    with pytest.raises(ValidationError, match="self-loop"):
        ScenarioSpec.model_validate(payload)


def test_warmup_cannot_exceed_half_the_horizon() -> None:
    with pytest.raises(ValidationError, match="warmup_days"):
        StressSpec(horizon_days=100.0, warmup_days=60.0)


def test_lane_requires_positive_distance() -> None:
    with pytest.raises(ValidationError):
        LaneSpec(a="A", b="B", distance_nm=0.0, zone="Z", cost_usd_per_teu=1.0, capacity_teu_day=1.0)


def test_network_detects_duplicate_port_ids(scenario: ScenarioSpec) -> None:
    payload = scenario.model_dump(mode="json")
    payload["network"]["ports"][1]["id"] = payload["network"]["ports"][0]["id"]
    with pytest.raises(ValidationError):
        NetworkSpec.model_validate(payload["network"])

"""Shared pytest fixtures for the maritime-sentinel suite."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from maritime_sentinel.config import DEFAULT_SCENARIO  # noqa: E402
from maritime_sentinel.schemas import ScenarioSpec  # noqa: E402


@pytest.fixture(scope="session")
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture()
def scenario() -> ScenarioSpec:
    return DEFAULT_SCENARIO


@pytest.fixture()
def short_scenario() -> ScenarioSpec:
    """A cheap 60-day scenario used by tests that do not need the full horizon."""

    stress = DEFAULT_SCENARIO.stress.model_copy(
        update={"horizon_days": 60.0, "warmup_days": 10.0, "capacity_scale": 1.30}
    )
    return DEFAULT_SCENARIO.model_copy(update={"stress": stress, "seed": 4242})

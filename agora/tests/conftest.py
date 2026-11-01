"""Shared fixtures for the ASEOWS test suite."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agora.model import AgoraModel  # noqa: E402
from agora.schemas import ExperimentSpec  # noqa: E402


@pytest.fixture(scope="session")
def default_spec() -> ExperimentSpec:
    return ExperimentSpec()


@pytest.fixture(scope="session")
def warm_model(default_spec: ExperimentSpec) -> AgoraModel:
    """One lattice advanced far enough for the market to have formed."""

    model = AgoraModel(default_spec)
    for _ in range(240):
        model.step()
    return model

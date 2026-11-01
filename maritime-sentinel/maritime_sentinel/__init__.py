"""MARITIME SENTINEL - autonomous macro-dynamic stress-testing lab.

Public surface:
    config          default Indonesian maritime scenario
    engine          SimPy discrete-event core
    montecarlo      Latin Hypercube perturbation driver
    metrics         KPI + convergence diagnostics
    graph           LangGraph self-healing build pipeline
"""

from __future__ import annotations

__version__ = "1.0.0"

from .config import DEFAULT_SCENARIO, load_scenario, scenario_dict
from .engine import RunResult, simulate
from .metrics import compute_metrics, summarize
from .montecarlo import MCResult, run_monte_carlo
from .schemas import ScenarioSpec

__all__ = [
    "__version__",
    "DEFAULT_SCENARIO",
    "load_scenario",
    "scenario_dict",
    "ScenarioSpec",
    "simulate",
    "RunResult",
    "run_monte_carlo",
    "MCResult",
    "compute_metrics",
    "summarize",
]

"""ASEOWS - emergent spatial-economic open-world simulator.

A Mesa agent-based lattice economy in which settlements specialise on their
territorial endowments, discover trading partners through a decentralised
gravity scan and pave the routes their own commerce wears down.
"""

from .market import MarketEngine, TradeEdge
from .metrics import DEFAULT_GATES, evaluate, run_study
from .model import AgoraModel
from .render import render_frame
from .schemas import (
    RESOURCES,
    Blueprint,
    Directive,
    EconomySpec,
    ExperimentSpec,
    FinalReport,
    GravitySpec,
    ShockSpec,
    ValidationReport,
    WorldSpec,
)
from .settlement import Settlement
from .world import generate_world

__version__ = "1.0.0"

__all__ = [
    "DEFAULT_GATES",
    "RESOURCES",
    "AgoraModel",
    "Blueprint",
    "Directive",
    "EconomySpec",
    "ExperimentSpec",
    "FinalReport",
    "GravitySpec",
    "MarketEngine",
    "Settlement",
    "ShockSpec",
    "TradeEdge",
    "ValidationReport",
    "WorldSpec",
    "evaluate",
    "generate_world",
    "render_frame",
    "run_study",
]

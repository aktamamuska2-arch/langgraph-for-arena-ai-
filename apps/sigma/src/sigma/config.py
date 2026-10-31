"""Configuration for the SIGMA platform.

Everything is driven by environment variables with deterministic defaults so a
run is reproducible without any external service (see PRD section 7).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[2]  # apps/sigma
REPO_ROOT = APP_ROOT.parent.parent  # repository root
DEFAULT_VAR_DIR = APP_ROOT / "var"


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    """Runtime settings. Frozen so a graph run cannot mutate global config."""

    # --- persistence -------------------------------------------------------
    db_path: Path = field(default_factory=lambda: Path(os.environ.get(
        "SIGMA_DB_PATH", str(DEFAULT_VAR_DIR / "checkpoints.sqlite"))))
    db_busy_timeout_ms: int = field(default_factory=lambda: _env_int("SIGMA_DB_BUSY_TIMEOUT_MS", 5_000))

    # --- graph limits (PRD NFR-4) -----------------------------------------
    max_revisions: int = field(default_factory=lambda: _env_int("SIGMA_MAX_REVISIONS", 3))
    max_tool_calls: int = field(default_factory=lambda: _env_int("SIGMA_MAX_TOOL_CALLS", 24))
    max_tokens: int = field(default_factory=lambda: _env_int("SIGMA_MAX_TOKENS", 120_000))
    recursion_limit: int = field(default_factory=lambda: _env_int("SIGMA_RECURSION_LIMIT", 60))

    # --- quality gate (PRD section 5) -------------------------------------
    acceptance_threshold: float = field(
        default_factory=lambda: _env_float("SIGMA_ACCEPTANCE_THRESHOLD", 0.78))
    max_sub_questions: int = field(default_factory=lambda: _env_int("SIGMA_MAX_SUB_QUESTIONS", 6))

    # --- state hygiene (PRD section 4.3) ----------------------------------
    evidence_cap: int = field(default_factory=lambda: _env_int("SIGMA_EVIDENCE_CAP", 200))
    messages_cap: int = field(default_factory=lambda: _env_int("SIGMA_MESSAGES_CAP", 120))
    tool_calls_cap: int = field(default_factory=lambda: _env_int("SIGMA_TOOL_CALLS_CAP", 200))
    audit_cap: int = field(default_factory=lambda: _env_int("SIGMA_AUDIT_CAP", 400))
    errors_cap: int = field(default_factory=lambda: _env_int("SIGMA_ERRORS_CAP", 50))
    evaluations_cap: int = field(default_factory=lambda: _env_int("SIGMA_EVALUATIONS_CAP", 12))
    context_window: int = field(default_factory=lambda: _env_int("SIGMA_CONTEXT_WINDOW", 24))

    # --- tools -------------------------------------------------------------
    tool_timeout_ms: int = field(default_factory=lambda: _env_int("SIGMA_TOOL_TIMEOUT_MS", 1_500))
    monte_carlo_paths: int = field(default_factory=lambda: _env_int("SIGMA_MC_PATHS", 4_000))

    # --- LLM provider ------------------------------------------------------
    llm_provider: str = field(default_factory=lambda: os.environ.get("SIGMA_LLM_PROVIDER", "deterministic"))
    llm_base_url: str = field(default_factory=lambda: os.environ.get("SIGMA_LLM_BASE_URL", ""))
    llm_api_key: str = field(default_factory=lambda: os.environ.get("SIGMA_LLM_API_KEY", ""))
    llm_model: str = field(default_factory=lambda: os.environ.get("SIGMA_LLM_MODEL", "sigma-mini"))
    llm_timeout_s: float = field(default_factory=lambda: _env_float("SIGMA_LLM_TIMEOUT_S", 20.0))

    # --- size budget (PRD section 10) --------------------------------------
    size_budget_mb: float = field(default_factory=lambda: _env_float("SIGMA_SIZE_BUDGET_MB", 125.0))

    def ensure_var_dir(self) -> Path:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        return self.db_path.parent


SETTINGS = Settings()

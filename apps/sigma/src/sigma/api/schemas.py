"""HTTP request/response models for the SIGMA API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from sigma.config import SETTINGS


class CreateSessionRequest(BaseModel):
    """Body of `POST /api/sessions`."""

    question: str = Field(min_length=3, max_length=600)
    seed: int = 7
    portfolio_id: str = "sigma-core"
    max_revisions: int | None = None
    acceptance_threshold: float | None = None
    hitl: bool = True

    model_config = {"json_schema_extra": {"example": {
        "question": "Bagaimana risiko dan alokasi modal portofolio kuartal ini?",
        "seed": 7,
        "acceptance_threshold": 0.78,
        "hitl": True,
    }}}


class ResumeRequest(BaseModel):
    """Body of `POST /api/sessions/{id}/resume` (the HITL answer)."""

    decision: str = Field(pattern="^(approve|revise|reject)$")
    note: str = ""
    reviewer: str = "human"


class ForkRequest(BaseModel):
    """Body of `POST /api/sessions/{id}/fork`."""

    checkpoint_id: str
    patch: dict[str, Any] | None = None


class SessionSummary(BaseModel):
    """One row of the session list."""

    session_id: str
    status: str
    question: str | None = None
    score: float | None = None
    revision: int = 0
    evidence: int = 0
    created_at: str | None = None
    awaiting_human: bool = False


class HealthResponse(BaseModel):
    """`GET /api/health` — also carries the size budget audit."""

    status: str = "ok"
    service: str = "sigma-analytics"
    version: str = "1.0.0"
    langgraph: str = ""
    provider: str = ""
    hitl: bool = True
    budget_mb: float = SETTINGS.size_budget_mb
    artifact_mb: float | None = None
    checkpoint_db_bytes: int = 0


__all__ = ["CreateSessionRequest", "ForkRequest", "HealthResponse",
           "ResumeRequest", "SessionSummary"]

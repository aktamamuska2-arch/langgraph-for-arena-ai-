"""Tool registry: schema, validation, bounded execution, deterministic results.

Tools are the only way an agent can touch the world. The registry enforces the
three properties the graph relies on:

* **validated** — arguments are parsed by a Pydantic model before execution,
* **bounded** — every call runs under a cooperative timeout (PRD section 9),
* **recorded** — every call yields an immutable `ToolCallRecord` for the audit trail.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from sigma.errors import ToolExecutionError, ToolTimeoutError
from sigma.state import Evidence, ToolCall, ToolCallRecord
from sigma.tools import market


class ToolContext(BaseModel):
    """Everything a tool is allowed to know about the current world."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    portfolio_id: str = "sigma-core"
    seed: int = 7
    horizon_quarters: int = 8
    monte_carlo_paths: int = 4_000
    state_evidence: dict[str, dict[str, Any]] = Field(default_factory=dict)


class ArgsModel(BaseModel):
    """Base for tool arguments: strict, so typos surface before execution."""

    model_config = ConfigDict(extra="forbid")


class QueryPortfolioArgs(ArgsModel):
    sector: str | None = None
    stage: str | None = None
    limit: int = Field(default=10, ge=1, le=50)


class ComputeRiskArgs(ArgsModel):
    venture_ids: list[str] = Field(default_factory=list, max_length=32)
    horizon_quarters: int = Field(default=8, ge=1, le=40)


class ScenarioArgs(ArgsModel):
    venture_ids: list[str] = Field(default_factory=list, max_length=32)
    paths: int = Field(default=4_000, ge=200, le=6_000)
    horizon_quarters: int = Field(default=8, ge=1, le=40)


class BenchmarkArgs(ArgsModel):
    sector: str


class EvidenceLookupArgs(ArgsModel):
    evidence_id: str


class ToolHandler(Protocol):
    def __call__(self, ctx: ToolContext, args: ArgsModel) -> dict[str, Any]: ...


@dataclass(frozen=True)
class ToolSpec:
    """A tool declaration: name, JSON schema, handler, and its budget cost."""

    name: str
    description: str
    args_model: type[ArgsModel]
    handler: ToolHandler
    token_cost: int = 120
    tags: tuple[str, ...] = field(default_factory=tuple)

    def json_schema(self) -> dict[str, Any]:
        """JSON Schema surfaced to the LLM and to `/api/tools`."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.args_model.model_json_schema(),
            "tags": list(self.tags),
        }


def _handle_query(ctx: ToolContext, args: ArgsModel) -> dict[str, Any]:
    assert isinstance(args, QueryPortfolioArgs)
    return market.query_portfolio(ctx.portfolio_id, ctx.seed, sector=args.sector,
                                  stage=args.stage, limit=args.limit)


def _handle_risk(ctx: ToolContext, args: ArgsModel) -> dict[str, Any]:
    assert isinstance(args, ComputeRiskArgs)
    return market.compute_risk_metrics(ctx.portfolio_id, ctx.seed,
                                       venture_ids=args.venture_ids,
                                       horizon_quarters=args.horizon_quarters)


def _handle_scenario(ctx: ToolContext, args: ArgsModel) -> dict[str, Any]:
    assert isinstance(args, ScenarioArgs)
    return market.scenario_simulate(ctx.portfolio_id, ctx.seed,
                                    venture_ids=args.venture_ids, paths=args.paths,
                                    horizon_quarters=args.horizon_quarters,
                                    monte_carlo_paths_cap=6_000)


def _handle_benchmark(ctx: ToolContext, args: ArgsModel) -> dict[str, Any]:
    assert isinstance(args, BenchmarkArgs)
    return market.benchmark_sector(ctx.portfolio_id, ctx.seed, sector=args.sector)


def _handle_evidence_lookup(ctx: ToolContext, args: ArgsModel) -> dict[str, Any]:
    assert isinstance(args, EvidenceLookupArgs)
    return market.evidence_lookup(ctx.portfolio_id, ctx.seed,
                                  evidence_id=args.evidence_id,
                                  state_evidence=ctx.state_evidence)


DEFAULT_TOOLS: tuple[ToolSpec, ...] = (
    ToolSpec("query_portfolio",
             "List ventures in the portfolio (optionally filtered) with capital aggregates.",
             QueryPortfolioArgs, _handle_query, tags=("discovery",)),
    ToolSpec("compute_risk_metrics",
             "Expected value, volatility, concentration (HHI) and a 0..1 risk score.",
             ComputeRiskArgs, _handle_risk, tags=("risk", "quant"), token_cost=180),
    ToolSpec("scenario_simulate",
             "Monte-Carlo terminal value distribution: P10/P50/P90, VaR95, probability of loss.",
             ScenarioArgs, _handle_scenario, tags=("risk", "simulation"), token_cost=320),
    ToolSpec("benchmark_sector",
             "Sector quartiles for growth and volatility, used to normalise a venture.",
             BenchmarkArgs, _handle_benchmark, tags=("context",)),
    ToolSpec("evidence_lookup",
             "Re-fetch an evidence item already collected (deduplication helper).",
             EvidenceLookupArgs, _handle_evidence_lookup, tags=("dedup",), token_cost=40),
)


@dataclass
class ToolResult:
    """Outcome of one bounded tool execution.

    `timed_out` is surfaced (not raised) so the node can decide whether to hand
    the failure to the runtime's retry policy — a timeout is transient, bad
    arguments are not.
    """

    record: ToolCallRecord
    payload: dict[str, Any]
    ok: bool
    timed_out: bool = False

    def to_evidence(self, sub_question_id: str | None = None, summary: str = "") -> Evidence:
        return Evidence(
            tool=self.record.tool,
            sub_question_id=sub_question_id,
            args=dict(self.record.args),
            payload=dict(self.payload),
            summary=summary or f"{self.record.tool}({_abbrev(self.record.args)})",
            provenance=f"tool:{self.record.tool}#{self.record.call_id}",
        )


def _abbrev(args: dict[str, Any], limit: int = 90) -> str:
    text = ",".join(f"{k}={v}" for k, v in args.items())
    return text if len(text) <= limit else text[: limit - 3] + "..."


class ToolRegistry:
    """Validates and executes tool calls under a cooperative timeout."""

    def __init__(self, tools: tuple[ToolSpec, ...] = DEFAULT_TOOLS,
                 *, timeout_ms: int = 1_500) -> None:
        self._tools: dict[str, ToolSpec] = {t.name: t for t in tools}
        self.timeout_ms = timeout_ms
        self._pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="sigma-tool")

    @property
    def names(self) -> list[str]:
        return sorted(self._tools)

    def spec(self, name: str) -> ToolSpec:
        try:
            return self._tools[name]
        except KeyError:
            raise ToolExecutionError(name, f"unknown tool; available: {sorted(self._tools)}") from None

    def schemas(self) -> list[dict[str, Any]]:
        return [self._tools[n].json_schema() for n in self.names]

    def validate(self, call: ToolCall) -> ArgsModel:
        spec = self.spec(call.tool)
        try:
            return spec.args_model.model_validate(call.args or {})
        except ValidationError as exc:
            raise ToolExecutionError(call.tool, f"invalid arguments: {_compact(exc)}") from None

    def execute(self, call: ToolCall, ctx: ToolContext) -> ToolResult:
        """Run one tool call, never raising for expected, classifiable failures."""
        started = time.perf_counter()
        try:
            spec = self.spec(call.tool)
            args = self.validate(call)
        except ToolExecutionError as exc:
            return ToolResult(
                record=ToolCallRecord(**call.model_dump(), ok=False, error=str(exc)),
                payload={"error": str(exc)}, ok=False)

        future = self._pool.submit(spec.handler, ctx, args)
        try:
            payload = future.result(timeout=self.timeout_ms / 1000.0)
        except TimeoutError as exc:  # concurrent.futures.TimeoutError == builtin in 3.11
            future.cancel()
            err = ToolTimeoutError(call.tool, self.timeout_ms)
            return ToolResult(
                record=ToolCallRecord(**call.model_dump(), ok=False, error=str(err),
                                      duration_ms=self.timeout_ms),
                payload={"error": str(err)}, ok=False, timed_out=True)
        except Exception as exc:  # noqa: BLE001 - classified, not swallowed
            return ToolResult(
                record=ToolCallRecord(**call.model_dump(), ok=False, error=f"{type(exc).__name__}: {exc}",
                                      duration_ms=int((time.perf_counter() - started) * 1000)),
                payload={"error": str(exc)}, ok=False)

        elapsed = int((time.perf_counter() - started) * 1000)
        ok = not str(payload.get("error", "") or "")
        return ToolResult(
            record=ToolCallRecord(**call.model_dump(), ok=ok, duration_ms=elapsed,
                                  error=str(payload["error"]) if not ok else None,
                                  output_digest=_digest(payload)),
            payload=payload, ok=ok)


def _digest(payload: dict[str, Any], limit: int = 6) -> dict[str, Any]:
    """A compact view of a tool result, safe to keep in the audit record."""
    out: dict[str, Any] = {}
    for i, (key, value) in enumerate(payload.items()):
        if i >= limit:
            out["_truncated"] = True
            break
        if isinstance(value, (int, float, str, bool)) or value is None:
            out[key] = value
        elif isinstance(value, list):
            out[key] = f"[{len(value)} items]"
        elif isinstance(value, dict):
            out[key] = {k: v for k, v in list(value.items())[:4]}
    return out


def _compact(exc: ValidationError) -> str:
    return "; ".join(f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}"
                     for e in exc.errors()[:3])

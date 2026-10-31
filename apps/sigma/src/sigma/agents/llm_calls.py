"""Bridge between nodes and the LLM provider.

Nodes ask for a *typed* result. The bridge handles the three things every
provider call needs: the transactional memory window, structured-output
validation, and a deterministic fallback when the model is unavailable or
returns something unusable (PRD section 7.1).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, TypeVar

from pydantic import BaseModel

from sigma.config import SETTINGS
from sigma.llm.base import CompletionRequest
from sigma.llm.factory import get_provider
from sigma.llm.memory import append, window
from sigma.state import AuditEvent, ScenarioContext, SigmaState, message

T = TypeVar("T", bound=BaseModel)


@dataclass
class StructuredResult:
    """What a node needs after a provider call."""

    value: Any
    update: dict[str, Any] = field(default_factory=dict)
    fallback_used: bool = False
    tokens: int = 0
    provider: str = "deterministic"

ROLE_SYSTEMS: dict[str, str] = {
    "planner": ("Anda adalah perencana analitik. Dekomposisi permintaan menjadi "
                "sub-pertanyaan terukur yang masing-masing dapat dijawab oleh satu tool."),
    "researcher": ("Anda adalah eksekutor riset. Pilih panggilan tool yang paling "
                   "informatif dan hindari mengulang panggilan yang identik."),
    "analyst": ("Anda adalah analis. Ringkas bukti menjadi rekomendasi aksi yang "
                "spesifik; angka hanya boleh berasal dari hasil tool."),
    "critic": ("Anda adalah evaluator. Nilai draf terhadap rubrik dan sebutkan gap "
               "yang konkret serta dapat ditindaklanjuti."),
}


def structured(
    state: SigmaState,
    *,
    role: str,
    response_model: type[T],
    fallback: Callable[[], T],
    tools: list[dict[str, Any]] | None = None,
    system: str | None = None,
    **extra_context: Any,
) -> StructuredResult:
    """Ask the provider for `response_model`; degrade to `fallback` on failure.

    The returned `update` carries the new transcript entries and, when a fallback
    happened, an audit event — so a node stays a declarative state transformation
    even though it talks to a model.
    """
    provider = get_provider()
    scenario: ScenarioContext | None = state.get("scenario")
    request = CompletionRequest(
        system=system or ROLE_SYSTEMS.get(role, "Anda adalah agen analitik SIGMA."),
        messages=window(state),
        tools=tools,
        response_model=response_model,
        context={**dict(state), **extra_context},
        seed=scenario.seed if scenario else 7,
    )

    fallback_used = False
    tokens = 0
    audit_extra: list[AuditEvent] = []
    try:
        response = provider.complete(request)
        if response.parsed is None:
            raise ValueError("provider returned no structured output")
        parsed = response.parsed
        tokens = token_budget(response.usage["total_tokens"])
        text = response.text
    except Exception as exc:  # noqa: BLE001 - any provider failure -> deterministic
        parsed = fallback()
        fallback_used = True
        tokens = 250
        text = f"[deterministic fallback: {type(exc).__name__}] {parsed.model_dump_json()}"
        audit_extra.append(_event(
            state, role, "provider_fallback",
            f"{type(exc).__name__}: {exc}"[:200],
            provider=provider.name, model=response_model.__name__))

    update = append(state, message("assistant", text[:1200], kind=role,
                                   provider=provider.name,
                                   fallback=fallback_used))
    if audit_extra:
        update.setdefault("audit_trail", []).extend(audit_extra)
    return StructuredResult(value=parsed, update=update, fallback_used=fallback_used,
                            tokens=tokens, provider=provider.name)


def _event(state: SigmaState, node: str, kind: str, message: str, **data: Any) -> AuditEvent:
    seq = (state.get("audit_trail") or [])[-1].seq + 1 if state.get("audit_trail") else 1
    return AuditEvent(seq=seq, node=node, kind=kind, message=message, data=data or {})


def token_budget(tokens: int) -> int:
    """Clamp a provider's token usage so a bad response cannot blow the budget."""
    return max(0, min(int(tokens), SETTINGS.max_tokens))


__all__ = ["ROLE_SYSTEMS", "StructuredResult", "structured", "token_budget"]

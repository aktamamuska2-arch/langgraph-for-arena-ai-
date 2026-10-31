"""LLM provider abstraction.

Nodes never import an SDK. They ask for a *structured* result and the provider
is responsible for producing it — from a deterministic offline engine, from an
HTTP chat endpoint, or from any LangChain chat model.

The abstraction exists for two reasons:

1. **Hermetic runs.** Without credentials the platform must still produce a
   complete, reproducible analysis (PRD NFR-3).
2. **Grounding.** Numbers are always computed by tools, never by the model; the
   model is asked for judgement, decomposition and narrative.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from pydantic import BaseModel


@dataclass
class CompletionRequest:
    """What a node wants from a model."""

    system: str
    messages: list[dict[str, Any]] = field(default_factory=list)
    tools: list[dict[str, Any]] | None = None
    response_model: type[BaseModel] | None = None
    context: dict[str, Any] | None = None  # graph state snapshot (deterministic engine)
    seed: int = 7
    temperature: float = 0.0


@dataclass
class LLMResponse:
    """What a node gets back."""

    text: str
    model: str
    provider: str
    parsed: BaseModel | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    fallback_used: bool = False

    @property
    def usage(self) -> dict[str, int]:
        return {"prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.prompt_tokens + self.completion_tokens}


class LLMProvider(Protocol):
    """Minimal contract every provider implements."""

    name: str

    def complete(self, request: CompletionRequest) -> LLMResponse: ...


__all__ = ["CompletionRequest", "LLMProvider", "LLMResponse"]

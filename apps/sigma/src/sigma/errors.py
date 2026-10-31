"""Error taxonomy for SIGMA.

Every failure is classified as `fatal` or `recoverable`; only that flag decides
whether the graph falls back to `containment` (PRD section 9).
"""

from __future__ import annotations


class SigmaError(Exception):
    """Base class. `fatal` decides routing: containment vs retry/next cycle."""

    fatal = False


class ToolExecutionError(SigmaError):
    """A tool call failed (bad arguments, timeout, internal error)."""

    fatal = False

    def __init__(self, tool: str, message: str) -> None:
        super().__init__(f"tool {tool!r} failed: {message}")
        self.tool = tool
        self.message = message


class ToolTimeoutError(ToolExecutionError):
    """A tool call exceeded its deadline."""

    def __init__(self, tool: str, timeout_ms: int) -> None:
        super().__init__(tool, f"exceeded {timeout_ms} ms")


class BudgetExhausted(SigmaError):
    """A run ran out of revisions / tool calls / tokens. Terminal but not fatal."""

    fatal = False


class ProviderError(SigmaError):
    """The LLM provider failed. Recoverable: nodes fall back to deterministic."""

    fatal = False


class InvalidResume(SigmaError):
    """The human sent a resume payload the graph cannot act on."""

    fatal = False

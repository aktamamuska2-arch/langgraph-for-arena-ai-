"""Optional bridge to any LangChain chat model.

`langchain-core` is already a LangGraph dependency, so this bridge costs zero
extra bytes while letting SIGMA run on any provider the host application has
configured (Bedrock, Vertex, Ollama, ...). It is never constructed implicitly:
the application must inject the model explicitly.
"""

from __future__ import annotations

import json
from typing import Any, Protocol

from pydantic import BaseModel

from sigma.errors import ProviderError
from sigma.llm.base import CompletionRequest, LLMResponse


class ChatModelLike(Protocol):
    """The subset of `BaseChatModel` this bridge needs."""

    def invoke(self, messages: list[Any]) -> Any: ...

    def with_structured_output(self, schema: type[BaseModel]) -> "ChatModelLike": ...


class LangChainChatProvider:
    """Adapts a LangChain chat model to the SIGMA provider interface."""

    name = "langchain"

    def __init__(self, model: ChatModelLike, *, model_name: str = "") -> None:
        self._model = model
        self.model_name = model_name or type(model).__name__

    def complete(self, request: CompletionRequest) -> LLMResponse:
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

        messages: list[Any] = [SystemMessage(content=request.system)]
        for item in request.messages:
            role = item.get("role", "user")
            content = item.get("content", "")
            messages.append(HumanMessage(content=content) if role == "user"
                            else AIMessage(content=content))

        try:
            if request.response_model is not None and hasattr(self._model,
                                                              "with_structured_output"):
                structured = self._model.with_structured_output(request.response_model)
                parsed = structured.invoke(messages)
                text = (parsed.model_dump_json() if isinstance(parsed, BaseModel)
                        else json.dumps(parsed, default=str))
            else:
                result = self._model.invoke(messages)
                text = getattr(result, "content", str(result))
                parsed = None
        except Exception as exc:  # noqa: BLE001 - provider failures are one class
            raise ProviderError(f"langchain model failed: {exc}") from exc

        return LLMResponse(text=text, model=self.model_name, provider=self.name,
                           parsed=parsed,
                           prompt_tokens=max(1, len(request.system) // 4),
                           completion_tokens=max(1, len(text) // 4))


__all__ = ["LangChainChatProvider"]

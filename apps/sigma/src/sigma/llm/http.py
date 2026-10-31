"""OpenAI-compatible HTTP provider.

Deliberately hand-rolled on `httpx` (already a transitive dependency) instead of
pulling a vendor SDK: a single SDK would add several megabytes to the artifact
budget for ~80 lines of HTTP (PRD section 10.4).
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ValidationError

from sigma.errors import ProviderError
from sigma.llm.base import CompletionRequest, LLMResponse


class HttpChatProvider:
    """Calls an OpenAI-compatible `/chat/completions` endpoint."""

    name = "http"

    def __init__(self, *, base_url: str, api_key: str = "", model: str = "sigma-mini",
                 timeout_s: float = 20.0, extra_headers: dict[str, str] | None = None) -> None:
        if not base_url:
            raise ProviderError("SIGMA_LLM_BASE_URL is not configured")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout_s = timeout_s
        self.extra_headers = extra_headers or {}

    def complete(self, request: CompletionRequest) -> LLMResponse:
        import httpx  # imported lazily: the offline path must not need it

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": request.system}, *request.messages],
            "temperature": request.temperature,
            "seed": request.seed,
        }
        if request.tools:
            payload["tools"] = [
                {"type": "function",
                 "function": {"name": t["name"], "description": t.get("description", ""),
                              "parameters": t.get("parameters", {})}}
                for t in request.tools
            ]
        if request.response_model is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": request.response_model.__name__,
                    "schema": request.response_model.model_json_schema(),
                    "strict": False,
                },
            }

        headers = {"Content-Type": "application/json", **self.extra_headers}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            response = httpx.post(f"{self.base_url}/chat/completions", json=payload,
                                  headers=headers, timeout=self.timeout_s)
            response.raise_for_status()
            body = response.json()
        except Exception as exc:  # noqa: BLE001 - network errors are one class here
            raise ProviderError(f"chat request failed: {exc}") from exc

        try:
            message = body["choices"][0]["message"]
            text = message.get("content") or json.dumps(message.get("tool_calls", []))
            usage = body.get("usage", {})
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(f"unexpected response shape: {body!r:.200}") from exc

        parsed = None
        if request.response_model is not None:
            parsed = _parse(request.response_model, text)
        return LLMResponse(
            text=text, model=body.get("model", self.model), provider=self.name,
            parsed=parsed,
            prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
            completion_tokens=int(usage.get("completion_tokens", 0) or 0),
        )


def _parse(model: type[BaseModel], text: str) -> BaseModel:
    """Validate model output, tolerating prose-wrapped JSON."""
    cleaned = text.strip()
    if cleaned.startswith("```"):  # models love fences
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.strip()
    try:
        return model.model_validate_json(cleaned)
    except ValidationError as exc:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start >= 0 and end > start:
            try:
                return model.model_validate_json(cleaned[start:end + 1])
            except ValidationError:
                pass
        raise ProviderError(f"model output does not match "
                            f"{model.__name__}: {exc.errors()[:2]}") from exc


__all__ = ["HttpChatProvider"]

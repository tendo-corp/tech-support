"""LLM engine over OpenRouter's OpenAI-compatible chat API, few-shot from the fixtures."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass

import httpx

from ..domain import Context, EngineResult, Message
from ..fewshot import FewShotSelector
from ..prompt import build_messages
from ..questionnaire import normalize_answers
from .base import EngineError

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)
DEFAULT_MODEL = "typesafe/jev-router"


@dataclass(frozen=True)
class OpenRouterConfig:
    api_key: str
    model: str
    # OpenRouter tries these in order when the primary model errors out.
    fallback_models: tuple[str, ...] = ()
    base_url: str = "https://openrouter.ai/api/v1"
    timeout: float = 40.0
    temperature: float = 0.0

    @classmethod
    def from_env(cls) -> "OpenRouterConfig":
        key = os.environ.get("OPEN_ROUTER_KEY") or os.environ.get("OPENROUTER_API_KEY") or ""
        if not key:
            raise EngineError("OPEN_ROUTER_KEY is not set")
        return cls(
            api_key=key,
            model=os.environ.get("CLASSIFIER_MODEL") or DEFAULT_MODEL,
            fallback_models=tuple(m.strip() for m in os.environ.get("FALLBACK_MODELS", "").split(",") if m.strip()),
            base_url=os.environ.get("OPENROUTER_BASE_URL", cls.base_url).rstrip("/"),
            timeout=float(os.environ.get("LLM_TIMEOUT", cls.timeout)),
        )


def parse_content(content: str) -> dict:
    """Model output -> dict; tolerates ```json fences and prose around the object."""
    match = _JSON_BLOCK.search(content or "")
    if not match:
        raise EngineError(f"no JSON in model output: {content[:200]!r}")
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise EngineError(f"invalid JSON in model output: {exc}") from exc
    if not isinstance(data, dict):
        raise EngineError("model output is not an object")
    return data


class LLMEngine:
    def __init__(self, config: OpenRouterConfig, selector: FewShotSelector | None = None,
                 client: httpx.Client | None = None):
        self.config = config
        self.selector = selector
        self.client = client or httpx.Client(timeout=config.timeout)
        self.name = f"llm:{config.model}"

    def answer(self, message: Message, context: Context | None = None) -> EngineResult:
        examples = self.selector.select(message) if self.selector else []
        body: dict = {
            "model": self.config.model,
            "messages": build_messages(message, examples, context),
            "temperature": self.config.temperature,
            "response_format": {"type": "json_object"},
        }
        if self.config.fallback_models:
            body["models"] = [self.config.model, *self.config.fallback_models]

        try:
            response = self.client.post(
                f"{self.config.base_url}/chat/completions",
                json=body,
                headers={"Authorization": f"Bearer {self.config.api_key}", "X-Title": "tech-support-classifier"},
            )
            response.raise_for_status()
            payload = response.json()
            content = payload["choices"][0]["message"]["content"]
        except (httpx.HTTPError, KeyError, IndexError, ValueError) as exc:
            raise EngineError(f"OpenRouter request failed: {exc}") from exc

        data = parse_content(content)
        raw_answers = data.get("answers", data)
        answers = normalize_answers(raw_answers if isinstance(raw_answers, dict) else {})
        if answers["is_task"] is None or answers["type"] is None:
            raise EngineError(f"model skipped required answers: {raw_answers!r}")
        return EngineResult(
            answers=answers,
            engine=f"llm:{payload.get('model', self.config.model)}",
            reasoning=str(data.get("reasoning", "")),
            confidence=0.85,
            raw={
                "usage": payload.get("usage"),
                "examples": [c.id for c in examples],
                "contract_item": data.get("contract_item") or None,
            },
        )

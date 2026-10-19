"""The classifier: JEV (typesafe/jev-router on OpenRouter) answers the questionnaire.

Few-shot examples come from the labelled fixtures; the reply is forced into the questionnaire's JSON schema.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass

import httpx

from .aggregate import to_verdict
from .domain import Context, EngineResult, Message, Verdict
from .fewshot import FewShotSelector
from .prompt import build_messages
from .questionnaire import json_schema, normalize_answers

DEFAULT_MODEL = "typesafe/jev-router"
# Statuses worth retrying: rate limit and upstream hiccups.
_RETRY_STATUS = {408, 429, 500, 502, 503, 504}


class JevError(RuntimeError):
    """JEV could not produce a usable answer."""


@dataclass(frozen=True)
class JevConfig:
    api_key: str
    model: str = DEFAULT_MODEL
    base_url: str = "https://openrouter.ai/api/v1"
    timeout: float = 60.0
    retries: int = 2
    # low | medium | high; None lets the router pick the effort per request.
    reasoning_effort: str | None = None

    @classmethod
    def from_env(cls) -> "JevConfig":
        key = os.environ.get("OPEN_ROUTER_KEY") or os.environ.get("OPENROUTER_API_KEY") or ""
        if not key:
            raise JevError("OPEN_ROUTER_KEY is not set")
        return cls(
            api_key=key,
            model=os.environ.get("JEV_MODEL") or DEFAULT_MODEL,
            base_url=os.environ.get("OPENROUTER_BASE_URL", cls.base_url).rstrip("/"),
            timeout=float(os.environ.get("JEV_TIMEOUT", cls.timeout)),
            reasoning_effort=os.environ.get("JEV_REASONING_EFFORT") or None,
        )


class Jev:
    def __init__(self, config: JevConfig, selector: FewShotSelector | None = None,
                 client: httpx.Client | None = None):
        self.config = config
        self.selector = selector
        self.client = client or httpx.Client(timeout=config.timeout)
        self.name = config.model

    def classify(self, message: Message, context: Context | None = None) -> Verdict:
        return to_verdict(self.answer(message, context))

    def answer(self, message: Message, context: Context | None = None) -> EngineResult:
        examples = self.selector.select(message) if self.selector else []
        body: dict = {
            "model": self.config.model,
            "messages": build_messages(message, examples, context),
            "temperature": 0,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "questionnaire", "strict": True, "schema": json_schema()},
            },
        }
        if self.config.reasoning_effort:
            body["reasoning"] = {"effort": self.config.reasoning_effort}

        started = time.monotonic()
        payload = self._post(body)
        try:
            data = json.loads(payload["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise JevError(f"unusable reply: {exc}") from exc

        answers = normalize_answers(data.get("answers") or {})
        if answers["is_task"] is None or answers["type"] is None:
            raise JevError(f"reply skipped required answers: {data!r}")

        item = data.get("contract_item") or None
        known = {i.id for i in context.contract.items} if context and context.contract else set()
        return EngineResult(
            answers=answers,
            engine=payload.get("model", self.config.model),
            reasoning=str(data.get("reasoning", "")),
            raw={
                # The model may invent an id; keep only ones that exist in the contract.
                "contract_item": item if item in known else None,
                "examples": [c.id for c in examples],
                "usage": payload.get("usage"),
                "seconds": round(time.monotonic() - started, 2),
            },
        )

    def _post(self, body: dict) -> dict:
        url = f"{self.config.base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self.config.api_key}", "X-Title": "tech-support-classifier"}
        last: Exception | None = None
        for attempt in range(self.config.retries + 1):
            if attempt:
                time.sleep(2 ** attempt)
            try:
                response = self.client.post(url, json=body, headers=headers)
            except httpx.TransportError as exc:
                last = exc
                continue
            if response.status_code in _RETRY_STATUS:
                last = JevError(f"HTTP {response.status_code}: {response.text[:200]}")
                continue
            if response.is_error:
                raise JevError(f"HTTP {response.status_code}: {response.text[:300]}")
            return response.json()
        raise JevError(f"OpenRouter unavailable after {self.config.retries + 1} attempts: {last}")

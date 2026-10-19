"""Labelled cases from fixtures/classifier/*.json — few-shot pool for the LLM and the eval set."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .domain import Answers, Attachment, Message
from .questionnaire import QUESTIONS, normalize_answers

DEFAULT_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "classifier"


class FixtureError(ValueError):
    pass


@dataclass(frozen=True)
class Case:
    id: str
    chat: str
    message: Message
    expected: Answers
    note: str = ""


def _parse_case(chat: str, raw: dict, source: Path) -> Case:
    where = f"{source.name}:{raw.get('id', '?')}"
    try:
        message = Message(
            text=raw["text"],
            author=raw.get("author", ""),
            author_side=raw.get("author_side", "client"),
            reply_to=raw.get("reply_to"),
            attachments=tuple(Attachment(kind=a["kind"], text=a.get("text", "")) for a in raw.get("attachments", [])),
        )
        given = raw["expected"]
    except KeyError as exc:
        raise FixtureError(f"{where}: missing field {exc}") from exc

    if message.author_side not in ("client", "team"):
        raise FixtureError(f"{where}: author_side must be client|team")
    unknown = set(given) - {q.id for q in QUESTIONS}
    if unknown:
        raise FixtureError(f"{where}: unknown questions {sorted(unknown)}")

    expected = normalize_answers(given)
    for q in QUESTIONS:
        applies = expected["is_task"] is True or not q.only_if_task
        if applies and expected[q.id] is None:
            raise FixtureError(f"{where}: '{q.id}' is missing or invalid ({given.get(q.id)!r})")
        if not applies and given.get(q.id) is not None:
            raise FixtureError(f"{where}: '{q.id}' must be null when is_task is false")
    return Case(id=raw["id"], chat=chat, message=message, expected=expected, note=raw.get("note", ""))


def load_cases(directory: Path = DEFAULT_DIR) -> list[Case]:
    cases: list[Case] = []
    for path in sorted(directory.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        cases.extend(_parse_case(data.get("chat", path.stem), raw, path) for raw in data["cases"])
    ids = [c.id for c in cases]
    duplicates = {i for i in ids if ids.count(i) > 1}
    if duplicates:
        raise FixtureError(f"duplicate case ids: {sorted(duplicates)}")
    return cases

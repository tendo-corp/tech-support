"""Labelled cases from fixtures/classifier/*.json — few-shot pool for the LLM and the eval set."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .contract import load_contracts
from .domain import Answers, Attachment, Context, Message
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
    context: Context | None = None
    # ClickUp task id the message is about (labelled when contract_match is A or B).
    contract_item: str | None = None


def _parse_case(chat: str, raw: dict, source: Path, context: Context | None = None) -> Case:
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
    item_id = raw.get("contract_item")
    if expected.get("contract_match") in ("A", "B"):
        items = {i.id: i for i in context.contract.items} if context and context.contract else {}
        if item_id not in items:
            raise FixtureError(f"{where}: contract_match={expected['contract_match']} needs a contract_item from the contract")
        if items[item_id].in_contract != (expected["contract_match"] == "A"):
            raise FixtureError(f"{where}: contract_match disagrees with billing of {item_id}")
    elif item_id is not None:
        raise FixtureError(f"{where}: contract_item set but contract_match is {expected.get('contract_match')!r}")
    return Case(id=raw["id"], chat=chat, message=message, expected=expected, note=raw.get("note", ""),
                context=context, contract_item=item_id)


def load_cases(directory: Path = DEFAULT_DIR) -> list[Case]:
    contracts = load_contracts(directory.parent / "contract")
    cases: list[Case] = []
    for path in sorted(directory.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        context = None
        if data.get("contract"):
            if data["contract"] not in contracts:
                raise FixtureError(f"{path.name}: unknown contract {data['contract']!r}")
            context = Context(project=data.get("chat", ""), contract=contracts[data["contract"]])
        cases.extend(_parse_case(data.get("chat", path.stem), raw, path, context) for raw in data["cases"])
    ids = [c.id for c in cases]
    duplicates = {i for i in ids if ids.count(i) > 1}
    if duplicates:
        raise FixtureError(f"duplicate case ids: {sorted(duplicates)}")
    return cases

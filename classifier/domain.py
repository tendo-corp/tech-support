"""Core types. No I/O here: the JEV client, storage and the bot depend on this module, not the other way round."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Union

AnswerValue = Union[bool, int, str, None]
Answers = dict[str, AnswerValue]


class Kind(str, Enum):
    BOOL = "bool"  # да / нет
    PERCENT = "percent"  # 0..100
    CHOICE = "choice"  # A / B / C / D / E


@dataclass(frozen=True)
class Question:
    id: str
    kind: Kind
    text: str
    # letter -> meaning, only for CHOICE
    options: dict[str, str] = field(default_factory=dict)
    # Asked only when the message is a task (is_task == True); otherwise the answer is None.
    only_if_task: bool = False

    def normalize(self, value: Any) -> AnswerValue:
        """Coerce a raw model/fixture value to the question's type; None if it doesn't fit."""
        if value is None:
            return None
        if self.kind is Kind.BOOL:
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                v = value.strip().lower()
                if v in ("да", "yes", "true", "1"):
                    return True
                if v in ("нет", "no", "false", "0"):
                    return False
            return None
        if self.kind is Kind.PERCENT:
            if isinstance(value, bool):
                return None
            try:
                number = float(str(value).strip().rstrip("%"))
            except ValueError:
                return None
            return int(round(min(100.0, max(0.0, number))))
        letter = str(value).strip().upper()[:1]
        return letter if letter in self.options else None


@dataclass(frozen=True)
class Attachment:
    kind: str  # photo | document | forward
    # What is visible in the attachment (OCR / vision / manual transcription). Empty if unknown.
    text: str = ""


@dataclass(frozen=True)
class Message:
    text: str
    author: str = ""
    # client = заказчик, team = мы (разработка/менеджмент)
    author_side: str = "client"
    reply_to: str | None = None
    attachments: tuple[Attachment, ...] = ()
    # Screenshots as data: URLs, sent to JEV as images (never Telegram file URLs: they contain the bot token).
    images: tuple[str, ...] = ()

    @property
    def full_text(self) -> str:
        parts = [self.text] + [a.text for a in self.attachments if a.text]
        return "\n".join(p for p in parts if p)


@dataclass(frozen=True)
class ContractItem:
    id: str  # ClickUp task id
    billing: str  # paid = входит в контракт, to_offer = не входит, предлагаем отдельно
    title: str
    summary: str = ""
    phrases: tuple[str, ...] = ()

    @property
    def in_contract(self) -> bool:
        return self.billing == "paid"


@dataclass(frozen=True)
class Contract:
    id: str
    project: str
    rules: tuple[str, ...] = ()
    items: tuple[ContractItem, ...] = ()


@dataclass(frozen=True)
class Context:
    project: str = ""
    # Free-text scope for customers without a formal contract file.
    contract_scope: str = ""
    contract: Contract | None = None


class Category(str, Enum):
    BUG = "bug"
    FEATURE = "feature"
    QUESTION = "question"
    DISCUSSION = "discussion"
    NOISE = "noise"


class Priority(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"


@dataclass
class EngineResult:
    answers: Answers
    engine: str
    reasoning: str = ""
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class Verdict:
    category: Category
    is_task: bool
    # True = входит в контракт (гарантийный баг или мелкая доработка), False = выход за рамки, None = не задача
    in_scope: bool | None
    # ClickUp task id the message is about, when the engine could tell.
    contract_item: str | None
    bug_probability: int | None
    estimate_hours: tuple[float, float | None] | None
    priority: Priority | None
    needs_clarification: bool | None
    answers: Answers
    engine: str
    reasoning: str = ""

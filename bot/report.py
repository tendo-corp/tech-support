"""Text of the report the bot posts to the internal chat."""

from __future__ import annotations

from html import escape

from classifier.domain import Contract, Verdict

ICON = {"bug": "🐞", "feature": "✨", "question": "❓", "discussion": "💬", "noise": "·"}
CATEGORY_RU = {"bug": "Баг", "feature": "Доработка", "question": "Вопрос", "discussion": "Обсуждение", "noise": "Шум"}
PRIORITY_RU = {"critical": "критично", "high": "высокий", "normal": "обычный", "low": "низкий"}


def _hours(verdict: Verdict) -> str | None:
    if not verdict.estimate_hours:
        return None
    low, high = verdict.estimate_hours
    return f"{low:g}–{high:g} ч" if high else f"от {low:g} ч"


def format_report(verdict: Verdict, *, chat: str, author: str, text: str, link: str | None,
                  contract: Contract | None) -> str:
    category = verdict.category.value
    scope = {True: "в рамках контракта", False: "<b>выход за рамки</b>", None: None}[verdict.in_scope]
    head = f"{ICON[category]} <b>{CATEGORY_RU[category]}</b>" + (f" · {scope}" if scope else "")
    quote = escape(text if len(text) <= 400 else text[:400] + "…")
    lines = [head, f"{escape(chat)} · {escape(author)}" + (f' · <a href="{link}">сообщение</a>' if link else ""),
             f"<blockquote>{quote}</blockquote>"]

    if verdict.contract_item and contract:
        item = next((i for i in contract.items if i.id == verdict.contract_item), None)
        if item:
            billing = "оплачено" if item.in_contract else "к предложению"
            lines.append(f'Задача: <a href="https://app.clickup.com/t/{item.id}">{escape(item.title)}</a> ({billing})')
    facts = [f for f in (
        f"оценка {_hours(verdict)}" if _hours(verdict) else None,
        f"приоритет {PRIORITY_RU[verdict.priority.value]}" if verdict.priority else None,
        "нужно уточнить у заказчика" if verdict.needs_clarification else None,
    ) if f]
    if facts:
        lines.append(" · ".join(facts))
    if verdict.reasoning:
        lines.append(f"<i>{escape(verdict.reasoning)}</i>")
    return "\n".join(lines)

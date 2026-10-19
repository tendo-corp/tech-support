"""Live check: classify a message with JEV and print the verdict.

    python -m classifier "При выгрузке листов КП они становятся заблокированными"
    python -m classifier                      # interactive: one message per line, empty line exits
    python -m classifier --team "Взяли в работу"
"""

from __future__ import annotations

import argparse
import sys

from .contract import load_contracts
from .dataset import load_cases
from .domain import Context, Message, Verdict
from .fewshot import FewShotSelector
from .jev import Jev, JevConfig, JevError
from .questionnaire import BY_ID, QUESTIONS

CATEGORY_RU = {"bug": "баг", "feature": "доработка", "question": "вопрос", "discussion": "обсуждение", "noise": "шум"}


def _hours(verdict: Verdict) -> str:
    if not verdict.estimate_hours:
        return "—"
    low, high = verdict.estimate_hours
    return f"{low:g}–{high:g} ч" if high else f"от {low:g} ч"


def format_verdict(verdict: Verdict, context: Context | None, seconds: float | None = None) -> str:
    scope = {True: "в рамках контракта", False: "ВЫХОД ЗА РАМКИ", None: "—"}[verdict.in_scope]
    item = ""
    if verdict.contract_item and context and context.contract:
        found = next(i for i in context.contract.items if i.id == verdict.contract_item)
        item = f"{found.title} [{found.billing}, {found.id}]"
    lines = [
        f"  итог:      {CATEGORY_RU[verdict.category.value]} · {scope}",
        f"  задача:    {item or '—'}",
        f"  оценка:    {_hours(verdict)} · приоритет {verdict.priority.value if verdict.priority else '—'}",
        f"  почему:    {verdict.reasoning}",
        "  анкета:",
    ]
    for q in QUESTIONS:
        value = verdict.answers.get(q.id)
        meaning = BY_ID[q.id].options.get(value, "") if isinstance(value, str) else ""
        lines.append(f"    {q.id:<20} {value!s:<6} {meaning}")
    if seconds is not None:
        lines.append(f"  ({seconds:.1f} с, {verdict.engine})")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("text", nargs="*", help="message text; omit for interactive mode")
    parser.add_argument("--team", action="store_true", help="message is from our team, not the customer")
    parser.add_argument("--reply", help="text of the message being replied to")
    parser.add_argument("--contract", default="aag_tenders", help="contract id from fixtures/contract ('' = none)")
    args = parser.parse_args(argv)

    contract = load_contracts().get(args.contract) if args.contract else None
    context = Context(contract=contract)
    try:
        jev = Jev(JevConfig.from_env(), selector=FewShotSelector(load_cases()))
    except JevError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    def run(text: str) -> None:
        message = Message(text=text, author_side="team" if args.team else "client", reply_to=args.reply)
        try:
            result = jev.answer(message, context)
        except JevError as exc:
            print(f"  ошибка JEV: {exc}")
            return
        from .aggregate import to_verdict

        print(format_verdict(to_verdict(result), context, result.raw.get("seconds")))

    if args.text:
        run(" ".join(args.text))
        return 0
    print("Сообщение от заказчика (пустая строка — выход):")
    for line in sys.stdin:
        if not line.strip():
            break
        run(line.strip())
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Prompt construction for LLM engines."""

from __future__ import annotations

import json
from collections.abc import Sequence

from .dataset import Case
from .domain import Answers, Context, Message
from .questionnaire import QUESTIONS, render

SYSTEM = """Ты — технический менеджер студии разработки. Ты разбираешь сообщения из рабочего чата с заказчиком.

Правило студии: ошибки в уже сделанном функционале исправляем в рамках контракта; любые доработки и пожелания \
разбираем отдельно — это выход за рамки, их оцениваем и согласуем. Мелкие правки (переименовать поле, убрать \
формулировку, скрыть элемент) можно сделать в рамках поддержки. Если заказчик спрашивает «почему так» или «как \
сделать» и за вопросом стоит неправильная работа системы — это баг, а не вопрос. Сообщения заказчика своим же \
коллегам, напоминания о статусе и ответы на уточнения — это обсуждение, а не задача.

Ответь на анкету по сообщению. Анкета:
{questions}

Верни только JSON вида {{"answers": {{"<id>": <значение>, ...}}, "reasoning": "<одно-два предложения>"}}.
Для вопросов с выбором отвечай одной буквой. Если is_task=false, вопросы с пометкой «только если is_task=true» = null."""


def format_message(message: Message) -> str:
    lines = [f"Автор: {message.author or 'неизвестно'} ({'заказчик' if message.author_side == 'client' else 'команда'})"]
    if message.reply_to:
        lines.append(f"В ответ на: «{message.reply_to}»")
    lines.append(f"Сообщение: {message.text}")
    for att in message.attachments:
        lines.append(f"Вложение ({att.kind}): {att.text or 'без описания'}")
    return "\n".join(lines)


def _answer_json(answers: Answers, note: str = "") -> str:
    payload = {"answers": {q.id: answers.get(q.id) for q in QUESTIONS}, "reasoning": note}
    return json.dumps(payload, ensure_ascii=False)


def build_messages(message: Message, examples: Sequence[Case], context: Context | None = None) -> list[dict]:
    system = SYSTEM.format(questions=render())
    if context and context.contract_scope:
        system += f"\n\nЧто входит в контракт с этим заказчиком:\n{context.contract_scope}"
    if context and context.project:
        system += f"\n\nПроект: {context.project}"

    chat = [{"role": "system", "content": system}]
    for case in examples:
        chat.append({"role": "user", "content": format_message(case.message)})
        chat.append({"role": "assistant", "content": _answer_json(case.expected, case.note)})
    chat.append({"role": "user", "content": format_message(message)})
    return chat

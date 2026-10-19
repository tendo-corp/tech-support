"""The questionnaire every engine answers. Fixtures are labelled against the same questions.

Changing an option letter's meaning invalidates the fixtures: add a new question id instead.
"""

from __future__ import annotations

from .domain import Answers, Kind, Question

IS_TASK = Question(
    id="is_task",
    kind=Kind.BOOL,
    text="Требует ли сообщение работы от команды разработки (исправить, сделать, изменить, расследовать)?",
)

TYPE = Question(
    id="type",
    kind=Kind.CHOICE,
    text="К какому типу относится сообщение?",
    options={
        "A": "баг — уже реализованный функционал работает неправильно",
        "B": "доработка — просьба добавить или изменить функционал",
        "C": "вопрос — как пользоваться, консультация",
        "D": "обсуждение — статус, координация, согласование, доступы, уточнение требований",
        "E": "шум — спасибо, ок, приветствие, реакция",
    },
)

BUG_PROBABILITY = Question(
    id="bug_probability",
    kind=Kind.PERCENT,
    text="Насколько вероятно (0–100), что сообщение описывает дефект уже реализованного функционала?",
)

COVERAGE = Question(
    id="coverage",
    kind=Kind.CHOICE,
    text="Как это соотносится с контрактом?",
    options={
        "A": "в рамках контракта — дефект в сделанном функционале или задача, уже оплаченная по договору (paid)",
        "B": "мелкая доработка (до ~2 ч) — можно сделать в рамках поддержки",
        "C": "выход за рамки — новая функциональность, расширение или задача «к предложению» (to_offer), нужна оценка",
        "D": "не задача",
    },
)

CONTRACT_MATCH = Question(
    id="contract_match",
    kind=Kind.CHOICE,
    text="К какой задаче из списка контракта относится сообщение?",
    options={
        "A": "к оплаченной задаче (paid) — входит в контракт",
        "B": "к задаче «к предложению» (to_offer) — не входит в контракт",
        "C": "ни к одной задаче из списка",
    },
    only_if_task=True,
)

COMPLEXITY = Question(
    id="complexity",
    kind=Kind.CHOICE,
    text="Сколько займёт работа?",
    options={
        "A": "до 1 часа",
        "B": "1–4 часа",
        "C": "1–2 дня",
        "D": "3–5 дней",
        "E": "больше недели, нужна декомпозиция",
    },
    only_if_task=True,
)

URGENCY = Question(
    id="urgency",
    kind=Kind.CHOICE,
    text="Насколько срочно?",
    options={
        "A": "критично — блокирует работу пользователей (не подать КП, данные теряются, доступ открыт посторонним)",
        "B": "высокая — мешает работать, есть обходной путь",
        "C": "обычная",
        "D": "низкая — косметика, можно отложить",
    },
    only_if_task=True,
)

NEEDS_CLARIFICATION = Question(
    id="needs_clarification",
    kind=Kind.BOOL,
    text="Нужно ли уточнить детали у заказчика, прежде чем браться за работу?",
    only_if_task=True,
)

QUESTIONS: tuple[Question, ...] = (
    IS_TASK,
    TYPE,
    BUG_PROBABILITY,
    COVERAGE,
    CONTRACT_MATCH,
    COMPLEXITY,
    URGENCY,
    NEEDS_CLARIFICATION,
)
BY_ID: dict[str, Question] = {q.id: q for q in QUESTIONS}


def normalize_answers(raw: dict) -> Answers:
    """Coerce every answer to its question's type and blank task-only answers for non-tasks."""
    answers: Answers = {q.id: q.normalize(raw.get(q.id)) for q in QUESTIONS}
    if answers["is_task"] is not True:
        for q in QUESTIONS:
            if q.only_if_task:
                answers[q.id] = None
    return answers


def render(questions: tuple[Question, ...] = QUESTIONS) -> str:
    """Human/LLM readable form of the questionnaire."""
    lines = []
    for q in questions:
        hint = {Kind.BOOL: "true/false", Kind.PERCENT: "целое 0–100", Kind.CHOICE: "одна буква"}[q.kind]
        suffix = " (только если is_task=true, иначе null)" if q.only_if_task else ""
        lines.append(f"- {q.id} [{hint}]{suffix}: {q.text}")
        for letter, meaning in q.options.items():
            lines.append(f"    {letter} — {meaning}")
    return "\n".join(lines)

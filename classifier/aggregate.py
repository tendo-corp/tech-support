"""Turns questionnaire answers into a business verdict."""

from __future__ import annotations

from .domain import Category, EngineResult, Priority, Verdict

CATEGORY_BY_TYPE = {
    "A": Category.BUG,
    "B": Category.FEATURE,
    "C": Category.QUESTION,
    "D": Category.DISCUSSION,
    "E": Category.NOISE,
}

# complexity letter -> (min, max) hours; None = open-ended
HOURS_BY_COMPLEXITY: dict[str, tuple[float, float | None]] = {
    "A": (0.25, 1),
    "B": (1, 4),
    "C": (8, 16),
    "D": (24, 40),
    "E": (40, None),
}

PRIORITY_BY_URGENCY = {
    "A": Priority.CRITICAL,
    "B": Priority.HIGH,
    "C": Priority.NORMAL,
    "D": Priority.LOW,
}


def to_verdict(result: EngineResult) -> Verdict:
    a = result.answers
    is_task = a.get("is_task") is True
    coverage = a.get("coverage")
    return Verdict(
        category=CATEGORY_BY_TYPE.get(a.get("type") or "", Category.DISCUSSION),
        is_task=is_task,
        in_scope=(coverage in ("A", "B")) if is_task and coverage in ("A", "B", "C") else None,
        bug_probability=a.get("bug_probability"),
        estimate_hours=HOURS_BY_COMPLEXITY.get(a.get("complexity") or ""),
        priority=PRIORITY_BY_URGENCY.get(a.get("urgency") or ""),
        needs_clarification=a.get("needs_clarification"),
        answers=a,
        engine=result.engine,
        reasoning=result.reasoning,
    )

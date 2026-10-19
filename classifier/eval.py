"""Scores an engine against the labelled fixtures.

    python -m classifier.eval                  # rules engine, offline
    python -m classifier.eval --engine llm -v  # OpenRouter, needs OPEN_ROUTER_KEY + CLASSIFIER_MODEL
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field

from .dataset import Case, load_cases
from .domain import AnswerValue, Kind
from .engines.base import Engine, EngineError
from .questionnaire import QUESTIONS

PERCENT_TOLERANCE = 20


def answers_match(kind: Kind, expected: AnswerValue, predicted: AnswerValue) -> bool:
    if kind is Kind.PERCENT and isinstance(expected, int) and isinstance(predicted, int):
        return abs(expected - predicted) <= PERCENT_TOLERANCE
    return expected == predicted


@dataclass
class Report:
    engine: str
    total: int = 0
    failed: list[str] = field(default_factory=list)
    correct: Counter = field(default_factory=Counter)
    asked: Counter = field(default_factory=Counter)
    # (expected type, predicted type) -> count
    confusion: Counter = field(default_factory=Counter)
    mistakes: list[tuple[str, str, AnswerValue, AnswerValue]] = field(default_factory=list)

    ITEM = "contract_item"

    def accuracy(self, question_id: str) -> float:
        return self.correct[question_id] / self.asked[question_id] if self.asked[question_id] else 0.0

    def format(self, verbose: bool = False) -> str:
        lines = [f"engine: {self.engine}   cases: {self.total}   errors: {len(self.failed)}", ""]
        for qid in [q.id for q in QUESTIONS] + [self.ITEM]:
            lines.append(f"  {qid:<20} {self.accuracy(qid):6.1%}  ({self.correct[qid]}/{self.asked[qid]})")
        lines += ["", "  type confusion (expected → predicted):"]
        for (exp, pred), n in sorted(self.confusion.items()):
            if exp != pred:
                lines.append(f"    {exp} → {pred}: {n}")
        if self.failed:
            lines += ["", "  engine errors: " + ", ".join(self.failed)]
        if verbose and self.mistakes:
            lines += ["", "  mistakes:"]
            lines += [f"    {cid:<7} {qid:<20} expected={exp!r} got={got!r}" for cid, qid, exp, got in self.mistakes]
        return "\n".join(lines)


def evaluate(engine: Engine, cases: Sequence[Case], client_only: bool = True) -> Report:
    report = Report(engine=engine.name)
    for case in cases:
        if client_only and case.message.author_side != "client":
            continue
        report.total += 1
        try:
            result = engine.answer(case.message, case.context)
            predicted = result.answers
        except EngineError as exc:
            report.failed.append(f"{case.id}: {exc}")
            continue
        report.confusion[(case.expected["type"], predicted.get("type"))] += 1
        for q in QUESTIONS:
            expected = case.expected[q.id]
            if expected is None:
                continue
            report.asked[q.id] += 1
            got = predicted.get(q.id)
            if answers_match(q.kind, expected, got):
                report.correct[q.id] += 1
            else:
                report.mistakes.append((case.id, q.id, expected, got))
        if case.contract_item:
            report.asked[Report.ITEM] += 1
            got_item = result.raw.get("contract_item")
            if got_item == case.contract_item:
                report.correct[Report.ITEM] += 1
            else:
                report.mistakes.append((case.id, Report.ITEM, case.contract_item, got_item))
    return report


def _build_engine(name: str, k: int) -> Engine:
    if name == "rules":
        from .engines.rules import RulesEngine

        return RulesEngine()
    from .engines.llm import LLMEngine, OpenRouterConfig
    from .fewshot import FewShotSelector

    return LLMEngine(OpenRouterConfig.from_env(), selector=FewShotSelector(load_cases(), k=k))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--engine", choices=("rules", "llm"), default="rules")
    parser.add_argument("--shots", type=int, default=8, help="few-shot examples per request (llm)")
    parser.add_argument("--only", help="comma-separated case ids")
    parser.add_argument("--include-team", action="store_true", help="also score team messages")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    cases = load_cases()
    if args.only:
        wanted = set(args.only.split(","))
        cases = [c for c in cases if c.id in wanted]
    try:
        engine = _build_engine(args.engine, args.shots)
    except EngineError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(evaluate(engine, cases, client_only=not args.include_team).format(verbose=args.verbose))
    return 0


if __name__ == "__main__":
    sys.exit(main())

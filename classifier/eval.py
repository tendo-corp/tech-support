"""Live eval: runs JEV over the labelled fixtures (leave-one-out few-shot) and scores every answer.

    OPEN_ROUTER_KEY=... python -m classifier.eval -v
    python -m classifier.eval --only ct-01,ct-26 -v
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from collections.abc import Sequence
from dataclasses import dataclass, field

from .dataset import Case, load_cases
from .domain import AnswerValue, EngineResult, Kind
from .fewshot import FewShotSelector
from .jev import Jev, JevConfig, JevError
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
    cost: float = 0.0
    # Which models the router picked, with spend per model.
    models: Counter = field(default_factory=Counter)
    model_cost: Counter = field(default_factory=Counter)
    seconds: list[float] = field(default_factory=list)

    ITEM = "contract_item"

    def accuracy(self, question_id: str) -> float:
        return self.correct[question_id] / self.asked[question_id] if self.asked[question_id] else 0.0

    def format(self, verbose: bool = False) -> str:
        lines = [f"model: {self.engine}   cases: {self.total}   errors: {len(self.failed)}"]
        if self.seconds:
            median = sorted(self.seconds)[len(self.seconds) // 2]
            lines.append(f"cost: ${self.cost:.4f}   latency median: {median:.1f}s   max: {max(self.seconds):.1f}s")
        for model, n in self.models.most_common():
            lines.append(f"  router → {model}: {n} req, ${self.model_cost[model]:.4f}")
        lines.append("")
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


def evaluate(jev: Jev, cases: Sequence[Case], workers: int = 4) -> Report:
    def run(case: Case) -> EngineResult | JevError:
        try:
            return jev.answer(case.message, case.context)
        except JevError as exc:
            return exc

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(run, cases))

    report = Report(engine=jev.name)
    for case, result in zip(cases, results):
        report.total += 1
        if isinstance(result, JevError):
            report.failed.append(f"{case.id}: {result}")
            continue
        cost = float((result.raw.get("usage") or {}).get("cost") or 0)
        report.cost += cost
        report.models[result.engine] += 1
        report.model_cost[result.engine] += cost
        report.seconds.append(result.raw.get("seconds", 0.0))
        predicted = result.answers
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


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--shots", type=int, default=8, help="few-shot examples per request")
    parser.add_argument("--only", help="comma-separated case ids")
    parser.add_argument("--clients-only", action="store_true", help="skip team messages")
    parser.add_argument("--workers", type=int, default=4, help="parallel requests")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    all_cases = load_cases()
    cases = all_cases
    if args.only:
        wanted = set(args.only.split(","))
        cases = [c for c in cases if c.id in wanted]
    if args.clients_only:
        cases = [c for c in cases if c.message.author_side == "client"]
    try:
        jev = Jev(JevConfig.from_env(), selector=FewShotSelector(all_cases, k=args.shots))
    except JevError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(evaluate(jev, cases, workers=args.workers).format(verbose=args.verbose))
    return 0


if __name__ == "__main__":
    sys.exit(main())

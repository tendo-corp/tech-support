"""Contract scope: which tasks are paid (in contract) and which are to offer (out of scope).

Source of truth is ClickUp (field Billing); fixtures/contract/*.json is a snapshot of it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .domain import Contract, ContractItem
from .text import similarity, stems

DEFAULT_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "contract"
BILLING = ("paid", "to_offer")


def parse_contract(data: dict) -> Contract:
    items = []
    for raw in data["items"]:
        if raw["billing"] not in BILLING:
            raise ValueError(f"contract {data['id']}: item {raw['id']} has billing {raw['billing']!r}")
        items.append(ContractItem(
            id=raw["id"], billing=raw["billing"], title=raw["title"],
            summary=raw.get("summary", ""), phrases=tuple(raw.get("phrases", ())),
        ))
    return Contract(id=data["id"], project=data.get("project", ""), rules=tuple(data.get("rules", ())), items=tuple(items))


def load_contracts(directory: Path = DEFAULT_DIR) -> dict[str, Contract]:
    contracts = [parse_contract(json.loads(p.read_text(encoding="utf-8"))) for p in sorted(directory.glob("*.json"))]
    return {c.id: c for c in contracts}


def render(contract: Contract) -> str:
    lines = [f"Проект: {contract.project}", "Правила:"]
    lines += [f"- {rule}" for rule in contract.rules]
    for billing, header in (("paid", "ВХОДИТ в контракт (paid)"), ("to_offer", "НЕ входит в контракт (to_offer)")):
        lines += ["", f"{header}:"]
        for item in contract.items:
            if item.billing == billing:
                lines.append(f"- [{item.id}] {item.title}. {item.summary}")
    return "\n".join(lines)


@dataclass(frozen=True)
class Match:
    item: ContractItem
    score: float


class ContractMatcher:
    """Bag-of-stems lookup of the contract item a message talks about. Crude, but free and offline."""

    def __init__(self, contract: Contract, threshold: float = 0.12):
        self.contract = contract
        self.threshold = threshold
        self._index = [
            (item, [stems(item.title)] + [stems(p) for p in item.phrases])
            for item in contract.items
        ]

    def match(self, text: str) -> Match | None:
        query = stems(text)
        best: Match | None = None
        for item, docs in self._index:
            score = max(similarity(query, d) for d in docs)
            if best is None or score > best.score:
                best = Match(item, score)
        return best if best and best.score >= self.threshold else None

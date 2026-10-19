"""Offline tests: JEV is replaced by a mock transport. Live checks: `python -m classifier.eval`."""

import json

import httpx
import pytest

from classifier.aggregate import to_verdict
from classifier.contract import load_contracts, render
from classifier.dataset import FixtureError, _parse_case, load_cases
from classifier.domain import Category, Context, EngineResult, Message, Priority
from classifier.fewshot import FewShotSelector
from classifier.jev import Jev, JevConfig, JevError
from classifier.prompt import build_messages
from classifier.questionnaire import BY_ID, QUESTIONS, json_schema, normalize_answers

CASES = load_cases()
BUG = {"is_task": True, "type": "A", "bug_probability": 90, "coverage": "A", "contract_match": "C",
       "complexity": "B", "urgency": "B", "needs_clarification": False}
CONTRACT = load_contracts()["aag_tenders"]


# --- fixtures & questionnaire -------------------------------------------------

def test_fixtures_load_and_cover_every_type():
    assert len(CASES) >= 80
    assert {c.expected["type"] for c in CASES} == {"A", "B", "C", "D", "E"}


def test_fixture_validation_rejects_bad_labels(tmp_path):
    with pytest.raises(FixtureError, match="coverage"):
        _parse_case("x", {"id": "x1", "text": "t", "expected": {"is_task": False, "type": "D", "bug_probability": 0}}, tmp_path / "f.json")
    with pytest.raises(FixtureError, match="must be null"):
        _parse_case("x", {"id": "x2", "text": "t", "expected": {"is_task": False, "type": "D", "bug_probability": 0,
                                                                "coverage": "D", "urgency": "A"}}, tmp_path / "f.json")


def test_fixture_contract_item_must_match_billing(tmp_path):
    raw = {"id": "x3", "text": "t", "contract_item": "869fbmt0r",  # to_offer item
           "expected": {**BUG, "contract_match": "A"}}
    with pytest.raises(FixtureError, match="billing"):
        _parse_case("x", raw, tmp_path / "f.json", Context(contract=CONTRACT))


@pytest.mark.parametrize("qid,raw,expected", [
    ("is_task", "да", True), ("is_task", "No", False), ("is_task", "maybe", None),
    ("bug_probability", "85%", 85), ("bug_probability", 140, 100), ("bug_probability", True, None),
    ("type", "b — доработка", "B"), ("type", "Z", None), ("coverage", "E", None),
])
def test_normalize(qid, raw, expected):
    assert BY_ID[qid].normalize(raw) == expected


def test_task_only_answers_are_blanked_for_non_tasks():
    answers = normalize_answers({"is_task": False, "type": "E", "urgency": "A", "complexity": "C"})
    assert answers["urgency"] is None and answers["complexity"] is None


def test_schema_covers_every_question():
    schema = json_schema()["properties"]["answers"]
    assert set(schema["required"]) == {q.id for q in QUESTIONS}
    assert schema["properties"]["type"]["enum"] == ["A", "B", "C", "D", "E"]
    assert {"type": "null"} in schema["properties"]["urgency"]["anyOf"]


def test_verdict_mapping():
    verdict = to_verdict(EngineResult(answers=normalize_answers({**BUG, "coverage": "C", "type": "B", "urgency": "A"}), engine="t"))
    assert verdict.category is Category.FEATURE
    assert verdict.in_scope is False
    assert verdict.priority is Priority.CRITICAL
    assert verdict.estimate_hours == (1, 4)

    noise = to_verdict(EngineResult(answers=normalize_answers({"is_task": False, "type": "E", "bug_probability": 0, "coverage": "D"}), engine="t"))
    assert noise.category is Category.NOISE and noise.in_scope is None and noise.estimate_hours is None


# --- contract & prompt --------------------------------------------------------

def test_contract_snapshot_has_both_billing_kinds():
    assert {i.billing for i in CONTRACT.items} == {"paid", "to_offer"}
    text = render(CONTRACT)
    assert "ВХОДИТ в контракт" in text and "НЕ входит" in text


def test_prompt_carries_contract_and_item_ids():
    case = next(c for c in CASES if c.contract_item)
    chat = build_messages(Message(text="что-то сломалось"), [case], case.context)
    assert "869fbmyhv" in chat[0]["content"]
    assert json.loads(chat[2]["content"])["contract_item"] == case.contract_item


# --- few-shot -----------------------------------------------------------------

def test_fewshot_excludes_the_query_and_covers_types():
    query = next(c for c in CASES if c.id == "tp-21").message
    chosen = FewShotSelector(CASES, k=8).select(query)
    assert len(chosen) == 8
    assert all(c.message.text != query.text for c in chosen)
    assert all(c.message.author_side == "client" for c in chosen)
    assert {c.expected["type"] for c in chosen} == {"A", "B", "C", "D", "E"}


def test_fewshot_for_team_messages_uses_team_examples():
    chosen = FewShotSelector(CASES, k=3).select(Message(text="Взяли в работу", author_side="team"))
    assert chosen and all(c.message.author_side == "team" for c in chosen)


# --- jev ----------------------------------------------------------------------

def _jev(handler, retries: int = 0) -> Jev:
    return Jev(JevConfig(api_key="k", retries=retries), selector=FewShotSelector(CASES, k=3),
               client=httpx.Client(transport=httpx.MockTransport(handler)))


def _reply(data: dict, model: str = "vendor/picked") -> httpx.Response:
    content = json.dumps(data, ensure_ascii=False)
    return httpx.Response(200, json={"model": model, "choices": [{"message": {"content": content}}],
                                     "usage": {"cost": 0.001}})


def test_jev_request_uses_strict_schema_and_parses_reply():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        seen["auth"] = request.headers["authorization"]
        return _reply({"answers": BUG, "contract_item": "869fbmyhv", "reasoning": "сломано ранжирование"})

    result = _jev(handler).answer(Message(text="В ранжировании только три участника"), Context(contract=CONTRACT))
    body = seen["body"]
    assert seen["auth"] == "Bearer k"
    assert body["model"] == "typesafe/jev-router"
    assert body["response_format"]["type"] == "json_schema"
    assert body["response_format"]["json_schema"]["strict"] is True
    assert len(body["messages"]) == 1 + 3 * 2 + 1  # system + few-shot pairs + query
    assert "869fbmyhv" in body["messages"][0]["content"]
    assert result.answers["type"] == "A"
    assert result.engine == "vendor/picked"
    assert result.raw["contract_item"] == "869fbmyhv"


def test_jev_drops_unknown_contract_item():
    jev = _jev(lambda r: _reply({"answers": BUG, "contract_item": "made-up", "reasoning": ""}))
    assert jev.answer(Message(text="что-то"), Context(contract=CONTRACT)).raw["contract_item"] is None


def test_jev_rejects_reply_without_type():
    jev = _jev(lambda r: _reply({"answers": {"is_task": True}, "contract_item": None, "reasoning": ""}))
    with pytest.raises(JevError):
        jev.answer(Message(text="что-то"))


def test_jev_retries_transient_errors(monkeypatch):
    monkeypatch.setattr("classifier.jev.time.sleep", lambda s: None)
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(503) if len(calls) == 1 else _reply({"answers": BUG, "contract_item": None, "reasoning": ""})

    assert _jev(handler, retries=1).classify(Message(text="упало")).category is Category.BUG
    assert len(calls) == 2


def test_jev_does_not_retry_client_errors():
    with pytest.raises(JevError, match="401"):
        _jev(lambda r: httpx.Response(401, text="bad key"), retries=3).answer(Message(text="x"))

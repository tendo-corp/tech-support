import json

import httpx
import pytest

from classifier.aggregate import to_verdict
from classifier.dataset import FixtureError, _parse_case, load_cases
from classifier.domain import Attachment, Category, EngineResult, Message, Priority
from classifier.engines.base import EngineError
from classifier.engines.llm import LLMEngine, OpenRouterConfig, parse_content
from classifier.engines.rules import RulesEngine
from classifier.eval import evaluate
from classifier.fewshot import FewShotSelector
from classifier.pipeline import Classifier
from classifier.questionnaire import BY_ID, normalize_answers

CASES = load_cases()
BUG = {"is_task": True, "type": "A", "bug_probability": 90, "coverage": "A",
       "complexity": "B", "urgency": "B", "needs_clarification": False}


# --- fixtures & questionnaire -------------------------------------------------

def test_fixtures_load_and_cover_every_type():
    assert len(CASES) >= 50
    assert {c.expected["type"] for c in CASES} == {"A", "B", "C", "D", "E"}


def test_fixture_validation_rejects_bad_labels(tmp_path):
    with pytest.raises(FixtureError, match="coverage"):
        _parse_case("x", {"id": "x1", "text": "t", "expected": {"is_task": False, "type": "D", "bug_probability": 0}}, tmp_path / "f.json")
    with pytest.raises(FixtureError, match="must be null"):
        _parse_case("x", {"id": "x2", "text": "t", "expected": {"is_task": False, "type": "D", "bug_probability": 0,
                                                                "coverage": "D", "urgency": "A"}}, tmp_path / "f.json")


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


def test_verdict_mapping():
    verdict = to_verdict(EngineResult(answers=normalize_answers({**BUG, "coverage": "C", "type": "B", "urgency": "A"}), engine="t"))
    assert verdict.category is Category.FEATURE
    assert verdict.in_scope is False
    assert verdict.priority is Priority.CRITICAL
    assert verdict.estimate_hours == (1, 4)

    noise = to_verdict(EngineResult(answers=normalize_answers({"is_task": False, "type": "E", "bug_probability": 0, "coverage": "D"}), engine="t"))
    assert noise.category is Category.NOISE and noise.in_scope is None and noise.estimate_hours is None


# --- rules engine -------------------------------------------------------------

@pytest.mark.parametrize("text", ["Спасибо!", "ок, спасибо", "да", "👍", "@monforje"])
def test_rules_noise(text):
    result = RulesEngine().answer(Message(text=text))
    assert result.answers["type"] == "E" and result.confidence >= 0.9


def test_rules_mention_with_attachment_is_not_noise():
    msg = Message(text="@monforje", attachments=(Attachment("forward", "Пытаюсь отправить КП, кнопка некликабельна"),))
    result = RulesEngine().answer(msg)
    assert result.answers["type"] == "A" and result.answers["urgency"] == "A"


def test_rules_team_messages_are_discussion():
    result = RulesEngine().answer(Message(text="Взяли в работу, чиним", author_side="team"))
    assert result.answers["type"] == "D" and result.answers["is_task"] is False


def test_rules_baseline_does_not_regress():
    report = evaluate(RulesEngine(), CASES)
    assert report.accuracy("type") >= 0.85
    assert report.accuracy("is_task") >= 0.85


# --- few-shot -----------------------------------------------------------------

def test_fewshot_excludes_the_query_and_covers_types():
    selector = FewShotSelector(CASES, k=8)
    query = next(c for c in CASES if c.id == "tp-21").message
    chosen = selector.select(query)
    assert len(chosen) == 8
    assert all(c.message.text != query.text for c in chosen)
    assert all(c.message.author_side == "client" for c in chosen)
    assert {c.expected["type"] for c in chosen} == {"A", "B", "C", "D", "E"}


# --- llm engine ---------------------------------------------------------------

def _engine(handler, **cfg) -> LLMEngine:
    config = OpenRouterConfig(api_key="k", model="vendor/model", **cfg)
    return LLMEngine(config, selector=FewShotSelector(CASES, k=3),
                     client=httpx.Client(transport=httpx.MockTransport(handler)))


def _reply(content: str, model: str = "vendor/model") -> httpx.Response:
    return httpx.Response(200, json={"model": model, "choices": [{"message": {"content": content}}], "usage": {"total_tokens": 1}})


def test_llm_request_and_parsing():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        seen["auth"] = request.headers["authorization"]
        return _reply("```json\n" + json.dumps({"answers": {**BUG, "type": "a", "bug_probability": "95"}, "reasoning": "сломана выгрузка"}) + "\n```")

    result = _engine(handler, fallback_models=("other/model",)).answer(Message(text="При выгрузке всё зависает"))
    body = seen["body"]
    assert seen["auth"] == "Bearer k"
    assert body["models"] == ["vendor/model", "other/model"]
    assert body["messages"][0]["role"] == "system" and "is_task" in body["messages"][0]["content"]
    assert len(body["messages"]) == 1 + 3 * 2 + 1  # system + few-shot pairs + query
    assert result.answers["type"] == "A" and result.answers["bug_probability"] == 95
    assert result.reasoning == "сломана выгрузка"
    assert len(result.raw["examples"]) == 3


def test_llm_rejects_answers_without_type():
    engine = _engine(lambda r: _reply(json.dumps({"answers": {"is_task": True}})))
    with pytest.raises(EngineError):
        engine.answer(Message(text="что-то"))


def test_parse_content_errors():
    with pytest.raises(EngineError):
        parse_content("no json here")


# --- pipeline -----------------------------------------------------------------

class _Boom:
    name = "boom"

    def answer(self, message, context=None):
        raise EngineError("down")


class _Spy:
    name = "spy"

    def __init__(self):
        self.calls = 0

    def answer(self, message, context=None):
        self.calls += 1
        return EngineResult(answers=normalize_answers(BUG), engine=self.name)


def test_pipeline_skips_llm_for_noise_and_team():
    spy = _Spy()
    clf = Classifier(smart=spy)
    assert clf.classify(Message(text="Спасибо!")).category is Category.NOISE
    assert clf.classify(Message(text="Починено", author_side="team")).category is Category.DISCUSSION
    assert spy.calls == 0
    assert clf.classify(Message(text="Не корректно считает")).engine == "spy"
    assert spy.calls == 1


def test_pipeline_falls_back_to_rules():
    verdict = Classifier(smart=_Boom()).classify(Message(text="При выгрузке листов КП они становятся заблокированными"))
    assert verdict.engine == "rules" and verdict.category is Category.BUG

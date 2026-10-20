import json

import httpx
import pytest

from bot.config import Settings
from bot.report import format_report
from bot.service import Bot
from bot.storage import MemoryStorage
from bot.updates import parse
from classifier.aggregate import to_verdict
from classifier.contract import load_contracts
from classifier.domain import EngineResult
from classifier.jev import Jev, JevConfig
from classifier.questionnaire import normalize_answers

CHAT = -1001234567890
TEAM_ID = 7
BUG = {"is_task": True, "type": "A", "bug_probability": 90, "coverage": "A", "contract_match": "A",
       "complexity": "B", "urgency": "B", "needs_clarification": False}


def update(text="", *, user_id=1, username="client", message_id=10, chat=CHAT, edited=False, **extra):
    msg = {"message_id": message_id, "date": 1760000000, "chat": {"id": chat, "title": "AAG & Tendo_ТП", "type": "supergroup"},
           "from": {"id": user_id, "is_bot": False, "first_name": "Влад", "username": username}, **extra}
    if text:
        msg["text"] = text
    return {"update_id": 1, ("edited_message" if edited else "message"): msg}


class FakeTelegram:
    def __init__(self):
        self.sent: list[tuple[int, str]] = []

    def send_message(self, chat_id, text, reply_to=None):
        self.sent.append((chat_id, text))

    def image_data_url(self, file_id):
        return "data:image/jpeg;base64,AAAA"


@pytest.fixture
def setup():
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        content = json.dumps({"answers": BUG, "contract_item": "869fbn5vy", "reasoning": "сломана загрузка Excel"})
        return httpx.Response(200, json={"model": "z-ai/glm-5.3-flash", "choices": [{"message": {"content": content}}],
                                         "usage": {"cost": 0.0003}})

    settings = Settings(telegram_token="t", webhook_secret="s", supabase_url="u", supabase_key="k",
                        notify_chat_id=-100999, team_user_ids=frozenset({TEAM_ID}))
    jev = Jev(JevConfig(api_key="k"), client=httpx.Client(transport=httpx.MockTransport(handler)))
    bot = Bot(settings, FakeTelegram(), MemoryStorage(), jev)
    return bot, requests


# --- parsing ------------------------------------------------------------------

def test_parse_photo_with_caption_reply_and_forward():
    inc = parse(update(caption="кнопка некликабельна", photo=[{"file_id": "small"}, {"file_id": "big"}],
                       reply_to_message={"message_id": 3, "text": "Взяли в работу"},
                       forward_origin={"type": "user", "sender_user": {"first_name": "Эдуард"}}))
    assert inc.text == "кнопка некликабельна"
    assert inc.photo_file_id == "big"
    assert inc.reply_to_text == "Взяли в работу" and inc.reply_to_message_id == 3
    assert inc.forwarded_from == "Эдуард"
    assert inc.link == "https://t.me/c/1234567890/10"


def test_parse_command_with_bot_mention():
    assert parse(update("/contract@tendo_tech_support_bot aag_tenders")).command == ("/contract", "aag_tenders")
    assert parse(update("обычный текст")).command is None


def test_parse_ignores_non_messages():
    assert parse({"update_id": 1, "my_chat_member": {}}) is None


# --- service ------------------------------------------------------------------

def test_customer_bug_is_classified_stored_and_reported(setup):
    bot, requests = setup
    assert bot.handle(update("Подрядчик загрузил Excel, а цены пустые")) == "classified: bug"

    row = bot.storage.messages[(CHAT, 10)]
    assert row["category"] == "bug" and row["in_scope"] is True
    assert row["contract_item"] == "869fbn5vy" and row["model"] == "z-ai/glm-5.3-flash"
    assert row["estimate_min_h"] == 1 and row["cost_usd"] == 0.0003
    # contract from DEFAULT_CONTRACT reaches the prompt
    assert "869fbn5vy" in requests[0]["messages"][0]["content"]

    (chat_id, report), = bot.telegram.sent
    assert chat_id == -100999
    assert "Баг" in report and "Подача КП из Excel" in report and "https://t.me/c/1234567890/10" in report


def test_screenshot_goes_to_jev_as_image(setup):
    bot, requests = setup
    bot.handle(update(caption="вот", photo=[{"file_id": "p"}]))
    content = requests[0]["messages"][-1]["content"]
    assert content[1] == {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,AAAA"}}


def test_team_messages_are_stored_not_classified(setup):
    bot, requests = setup
    assert bot.handle(update("Взяли в работу", user_id=TEAM_ID)) == "stored: team"
    assert requests == [] and bot.storage.messages[(CHAT, 10)]["author_side"] == "team"


def test_redelivered_update_is_not_reclassified(setup):
    bot, requests = setup
    bot.handle(update("Не корректно считает"))
    assert bot.handle(update("Не корректно считает")) == "skipped: already classified"
    assert len(requests) == 1


def test_edited_message_is_reclassified(setup):
    bot, requests = setup
    bot.handle(update("Не корректно считает"))
    bot.handle(update("Не корректно считает процент заполнения", edited=True))
    assert len(requests) == 2 and bot.storage.messages[(CHAT, 10)]["edited"] is True


def test_noise_is_stored_but_not_reported(setup):
    bot, _ = setup
    bot.jev.client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, json={
        "model": "m", "choices": [{"message": {"content": json.dumps({"answers": {
            "is_task": False, "type": "E", "bug_probability": 0, "coverage": "D"}, "contract_item": None, "reasoning": ""})}}]})))
    assert bot.handle(update("Спасибо!")) == "classified: noise"
    assert bot.telegram.sent == []


def test_jev_failure_is_recorded(setup):
    bot, _ = setup
    bot.jev.client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(401, text="bad key")))
    assert bot.handle(update("упало всё")) == "error: jev"
    assert "401" in bot.storage.messages[(CHAT, 10)]["error"]


def test_contract_command_is_team_only(setup):
    bot, _ = setup
    assert bot.handle(update("/contract aag_tenders")) == "ignored: command from non-team"
    assert bot.handle(update("/contract aag_tenders", user_id=TEAM_ID)) == "command: contract set"
    assert bot.storage.chats[CHAT]["contract_id"] == "aag_tenders"
    assert bot.handle(update("/contract nope", user_id=TEAM_ID)) == "command: contract unknown"


def test_notify_chat_itself_is_ignored(setup):
    bot, requests = setup
    assert bot.handle(update("Баг!", chat=-100999)) == "ignored: notify chat"
    assert requests == []


def test_report_escapes_html():
    verdict = to_verdict(EngineResult(answers=normalize_answers(BUG), engine="m", reasoning="<b>x</b>"))
    text = format_report(verdict, chat="A&B", author="<script>", text="1 < 2", link=None,
                         contract=load_contracts()["aag_tenders"])
    assert "&lt;script&gt;" in text and "1 &lt; 2" in text and "A&amp;B" in text and "&lt;b&gt;x" in text


def test_echo_mode_replies_in_source_chat(setup):
    from dataclasses import replace

    bot, _ = setup
    bot.settings = replace(bot.settings, echo_reports=True, notify_chat_id=None)
    bot.handle(update("Подрядчик загрузил Excel, а цены пустые"))
    assert [chat for chat, _ in bot.telegram.sent] == [CHAT]

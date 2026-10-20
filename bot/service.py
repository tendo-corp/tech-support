"""What happens to one Telegram update: store it, classify customer messages with JEV, report to the team."""

from __future__ import annotations

import logging
from html import escape

from classifier.aggregate import to_verdict
from classifier.contract import load_contracts
from classifier.dataset import load_cases
from classifier.domain import Attachment, Category, Context, Message, Verdict
from classifier.fewshot import FewShotSelector
from classifier.jev import Jev, JevConfig, JevError

from .config import Settings
from .report import format_report
from .storage import Storage
from .telegram import Telegram, TelegramError
from .updates import Incoming, parse

log = logging.getLogger(__name__)

HELP = """Я читаю чат с заказчиком и разбираю сообщения заказчиков: баг или выход за рамки контракта, сколько займёт, насколько срочно. Разбор присылаю в личку.

Команды:
/contract — какой контракт привязан к чату
/contract &lt;id&gt; — привязать контракт ({contracts})

В личке: напиши любое сообщение «от заказчика» — пришлю разбор, чтобы проверить."""

SUBSCRIBED = "Готово, буду присылать сюда разбор сообщений от: {clients}.\nМожешь проверить — напиши сюда любое сообщение «от заказчика»."


class Bot:
    def __init__(self, settings: Settings, telegram: Telegram, storage: Storage, jev: Jev):
        self.settings = settings
        self.telegram = telegram
        self.storage = storage
        self.jev = jev
        self.contracts = load_contracts()

    @classmethod
    def from_env(cls) -> "Bot":
        settings = Settings.from_env()
        return cls(
            settings,
            Telegram(settings.telegram_token),
            Storage(settings.supabase_url, settings.supabase_key),
            Jev(JevConfig.from_env(), selector=FewShotSelector(load_cases())),
        )

    def handle(self, update: dict) -> str:
        """Returns a short status for logs."""
        inc = parse(update)
        if inc is None:
            return "ignored: not a message"
        if inc.is_bot:
            return "ignored: bot"
        if inc.is_private:
            return self._private(inc)
        if inc.command:
            return self._command(inc)

        self.storage.upsert_chat(inc.chat_id, inc.chat_title)
        is_client = self.settings.is_client(inc.username)
        row = self._row(inc, "client" if is_client else "team")
        if not is_client or not (inc.text or inc.photo_file_id):
            self.storage.upsert_message(row)
            return "stored: not a client" if not is_client else "stored: no text"

        existing = self.storage.get_message(inc.chat_id, inc.message_id)
        if existing and existing.get("category") and existing.get("text") == inc.text:
            return "skipped: already classified"  # Telegram re-delivered the update

        contract = self._contract(inc.chat_id)
        try:
            result = self.jev.answer(self._message(inc), Context(project=inc.chat_title, contract=contract))
        except JevError as exc:
            log.error("JEV failed for %s/%s: %s", inc.chat_id, inc.message_id, exc)
            self.storage.upsert_message({**row, "error": str(exc)[:1000]})
            return "error: jev"

        verdict = to_verdict(result)
        self.storage.upsert_message({**row, **_verdict_columns(verdict, result.raw), "error": None})
        if verdict.category is not Category.NOISE:
            report = format_report(verdict, chat=inc.chat_title, author=_who(inc), text=inc.text,
                                   link=inc.link, contract=contract)
            for chat_id in self.storage.recipient_chats():
                self._send(chat_id, report)
        return f"classified: {verdict.category.value}"

    def _private(self, inc: Incoming) -> str:
        """Private chat: only recipients. Any message subscribes them; plain text is a test drive."""
        if not self.settings.is_recipient(inc.username):
            return "ignored: private from stranger"
        self.storage.upsert_recipient(inc.username, inc.chat_id)
        if inc.command:
            if inc.command[0] in ("/start", "/help"):
                clients = ", ".join(f"@{c}" for c in sorted(self.settings.client_usernames)) or "—"
                self._send(inc.chat_id, SUBSCRIBED.format(clients=clients))
                return "subscribed"
            return self._command(inc)
        if not (inc.text or inc.photo_file_id):
            return "ignored: empty"
        contract = self._contract(inc.chat_id)
        try:
            result = self.jev.answer(self._message(inc), Context(project="проверка в личке", contract=contract))
        except JevError as exc:
            self._send(inc.chat_id, f"JEV не ответил: {escape(str(exc)[:300])}", inc.message_id)
            return "error: jev"
        verdict = to_verdict(result)
        self._send(inc.chat_id, format_report(verdict, chat="проверка", author=_who(inc), text=inc.text,
                                              link=None, contract=contract), inc.message_id)
        return f"test: {verdict.category.value}"

    # --- helpers ---------------------------------------------------------------

    def _message(self, inc: Incoming) -> Message:
        attachments = []
        if inc.forwarded_from:
            attachments.append(Attachment("forward", f"переслано от {inc.forwarded_from}"))
        if inc.document_name:
            attachments.append(Attachment("document", inc.document_name))
        images: tuple[str, ...] = ()
        if inc.photo_file_id:
            try:
                url = self.telegram.image_data_url(inc.photo_file_id)
                images = (url,) if url else ()
            except Exception as exc:  # a broken photo must not block classification
                log.warning("photo download failed: %s", exc)
            if not images:
                attachments.append(Attachment("photo", "скриншот (не удалось загрузить)"))
        return Message(text=inc.text or "(без текста, только скриншот)", author=inc.author, author_side="client",
                       reply_to=inc.reply_to_text, attachments=tuple(attachments), images=images)

    def _row(self, inc: Incoming, side: str) -> dict:
        return {
            "chat_id": inc.chat_id, "message_id": inc.message_id, "thread_id": inc.thread_id,
            "user_id": inc.user_id, "username": inc.username, "author": inc.author, "author_side": side,
            "text": inc.text, "reply_to_message_id": inc.reply_to_message_id, "reply_to_text": inc.reply_to_text,
            "has_photo": bool(inc.photo_file_id), "sent_at": inc.sent_at, "edited": inc.edited,
        }

    def _contract(self, chat_id: int):
        return self.contracts.get(self.storage.chat_contract(chat_id) or self.settings.default_contract)

    def _send(self, chat_id: int, text: str, reply_to: int | None = None) -> None:
        try:
            self.telegram.send_message(chat_id, text, reply_to)
        except TelegramError as exc:
            log.error("send to %s failed: %s", chat_id, exc)

    def _command(self, inc: Incoming) -> str:
        name, arg = inc.command
        if name in ("/start", "/help"):
            self._send(inc.chat_id, HELP.format(contracts=", ".join(self.contracts)), inc.message_id)
            return "command: help"
        if not self.settings.is_recipient(inc.username):
            return "ignored: command from non-recipient"
        if name == "/contract":
            if arg:
                if arg not in self.contracts:
                    self._send(inc.chat_id, f"Нет контракта «{escape(arg)}». Есть: {', '.join(self.contracts)}", inc.message_id)
                    return "command: contract unknown"
                self.storage.upsert_chat(inc.chat_id, inc.chat_title, arg)
                self._send(inc.chat_id, f"Контракт чата: <b>{escape(arg)}</b>", inc.message_id)
                return "command: contract set"
            current = self.storage.chat_contract(inc.chat_id) or f"{self.settings.default_contract} (по умолчанию)"
            self._send(inc.chat_id, f"Контракт чата: <b>{escape(current)}</b>", inc.message_id)
            return "command: contract show"
        return "ignored: unknown command"


def _who(inc: Incoming) -> str:
    return f"{inc.author} (@{inc.username})" if inc.username else inc.author


def _verdict_columns(verdict: Verdict, raw: dict) -> dict:
    a = verdict.answers
    low, high = verdict.estimate_hours or (None, None)
    return {
        "category": verdict.category.value,
        "is_task": verdict.is_task,
        "in_scope": verdict.in_scope,
        "contract_item": verdict.contract_item,
        "bug_probability": verdict.bug_probability,
        "complexity": a.get("complexity"),
        "urgency": a.get("urgency"),
        "estimate_min_h": low,
        "estimate_max_h": high,
        "priority": verdict.priority.value if verdict.priority else None,
        "needs_clarification": verdict.needs_clarification,
        "answers": a,
        "reasoning": verdict.reasoning,
        "model": verdict.engine,
        "cost_usd": (raw.get("usage") or {}).get("cost"),
        "latency_s": raw.get("seconds"),
    }

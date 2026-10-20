"""Telegram update JSON -> the fields the bot needs. Pure functions, no I/O."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass(frozen=True)
class Incoming:
    chat_id: int
    chat_title: str
    message_id: int
    thread_id: int | None
    user_id: int | None
    username: str | None
    author: str
    is_bot: bool
    text: str  # text or caption
    reply_to_text: str | None
    reply_to_message_id: int | None
    photo_file_id: str | None
    document_name: str | None
    forwarded_from: str | None
    sent_at: str
    edited: bool

    @property
    def command(self) -> tuple[str, str] | None:
        """('/contract', 'aag_tenders') for '/contract@bot aag_tenders'; None for ordinary text."""
        if not self.text.startswith("/"):
            return None
        head, _, rest = self.text.partition(" ")
        return head.split("@")[0].lower(), rest.strip()

    @property
    def link(self) -> str | None:
        # Supergroup ids look like -100XXXXXXXXXX; public t.me/c links use the XXXXXXXXXX part.
        raw = str(self.chat_id)
        return f"https://t.me/c/{raw[4:]}/{self.message_id}" if raw.startswith("-100") else None


def _name(user: dict | None) -> str:
    if not user:
        return ""
    return " ".join(p for p in (user.get("first_name"), user.get("last_name")) if p) or user.get("username", "")


def parse(update: dict) -> Incoming | None:
    edited = "edited_message" in update
    msg = update.get("edited_message") if edited else update.get("message")
    if not msg or "chat" not in msg:
        return None
    user = msg.get("from") or {}
    reply = msg.get("reply_to_message") or {}
    photos = msg.get("photo") or []
    document = msg.get("document") or {}
    # Photos sent "as file" arrive as documents with an image mime type.
    if not photos and str(document.get("mime_type", "")).startswith("image/"):
        photo_id = document.get("file_id")
    else:
        photo_id = photos[-1]["file_id"] if photos else None  # last = largest size
    origin = msg.get("forward_origin") or {}
    forwarded = _name(origin.get("sender_user")) or origin.get("sender_user_name") or (origin.get("chat") or {}).get("title")
    return Incoming(
        chat_id=msg["chat"]["id"],
        chat_title=msg["chat"].get("title") or _name(msg["chat"]),
        message_id=msg["message_id"],
        thread_id=msg.get("message_thread_id"),
        user_id=user.get("id"),
        username=user.get("username"),
        author=_name(user),
        is_bot=bool(user.get("is_bot")),
        text=(msg.get("text") or msg.get("caption") or "").strip(),
        reply_to_text=(reply.get("text") or reply.get("caption") or None),
        reply_to_message_id=reply.get("message_id"),
        photo_file_id=photo_id,
        document_name=None if photo_id == document.get("file_id") else document.get("file_name"),
        forwarded_from=forwarded or None,
        sent_at=datetime.fromtimestamp(msg.get("edit_date") or msg["date"], tz=timezone.utc).isoformat(),
        edited=edited,
    )

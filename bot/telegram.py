"""Minimal Telegram Bot API client: only what the bot uses."""

from __future__ import annotations

import base64

import httpx

API = "https://api.telegram.org"
# Bigger photos are skipped rather than sent to the model.
MAX_IMAGE_BYTES = 4 * 1024 * 1024


class TelegramError(RuntimeError):
    pass


class Telegram:
    def __init__(self, token: str, client: httpx.Client | None = None):
        self.token = token
        self.client = client or httpx.Client(timeout=20)

    def call(self, method: str, **params) -> dict | list | bool:
        response = self.client.post(f"{API}/bot{self.token}/{method}", json=params)
        data = response.json()
        if not data.get("ok"):
            raise TelegramError(f"{method}: {data.get('description', response.status_code)}")
        return data["result"]

    def send_message(self, chat_id: int, text: str, reply_to: int | None = None) -> None:
        params: dict = {"chat_id": chat_id, "text": text, "parse_mode": "HTML", "link_preview_options": {"is_disabled": True}}
        if reply_to:
            params["reply_parameters"] = {"message_id": reply_to, "allow_sending_without_reply": True}
        self.call("sendMessage", **params)

    def image_data_url(self, file_id: str) -> str | None:
        """Download a photo and return it as a data: URL (keeps the token out of anything sent to the model)."""
        info = self.call("getFile", file_id=file_id)
        if info.get("file_size", 0) > MAX_IMAGE_BYTES or not info.get("file_path"):
            return None
        response = self.client.get(f"{API}/file/bot{self.token}/{info['file_path']}")
        response.raise_for_status()
        mime = "image/png" if info["file_path"].lower().endswith(".png") else "image/jpeg"
        return f"data:{mime};base64,{base64.b64encode(response.content).decode()}"

    def set_webhook(self, url: str, secret: str) -> None:
        self.call("setWebhook", url=url, secret_token=secret,
                  allowed_updates=["message", "edited_message"], drop_pending_updates=False)

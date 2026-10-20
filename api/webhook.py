"""Vercel serverless entry point: POST /api/webhook receives Telegram updates."""

from __future__ import annotations

import hmac
import json
import logging
import sys
from http.server import BaseHTTPRequestHandler
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bot.config import Settings  # noqa: E402
from bot.service import Bot  # noqa: E402

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("webhook")

_bot: Bot | None = None  # reused across invocations of a warm function


def get_bot() -> Bot:
    global _bot
    if _bot is None:
        _bot = Bot.from_env()
    return _bot


class handler(BaseHTTPRequestHandler):
    def _send(self, status: int, body: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.end_headers()
        self.wfile.write(body.encode())

    def do_GET(self) -> None:
        missing = Settings.from_env().missing()
        self._send(500 if missing else 200, f"missing env: {', '.join(missing)}" if missing else "ok")

    def do_POST(self) -> None:
        settings = Settings.from_env()
        secret = self.headers.get("X-Telegram-Bot-Api-Secret-Token", "")
        if not settings.webhook_secret or not hmac.compare_digest(secret, settings.webhook_secret):
            self._send(401, "unauthorized")
            return
        try:
            update = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        except json.JSONDecodeError:
            self._send(400, "bad json")
            return
        try:
            status = get_bot().handle(update)
        except Exception:  # never make Telegram retry forever on our bug
            log.exception("update %s failed", update.get("update_id"))
            status = "error"
        log.info("update %s: %s", update.get("update_id"), status)
        self._send(200, status)

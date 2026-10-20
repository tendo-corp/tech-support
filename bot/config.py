"""Bot settings from environment variables (Vercel project env)."""

from __future__ import annotations

import os
from dataclasses import dataclass


def _names(raw: str) -> frozenset[str]:
    return frozenset(x.strip().lstrip("@").lower() for x in raw.split(",") if x.strip())


@dataclass(frozen=True)
class Settings:
    telegram_token: str
    # Telegram sends it in X-Telegram-Bot-Api-Secret-Token; requests without it are rejected.
    webhook_secret: str
    supabase_url: str
    supabase_key: str
    # Customers whose messages get classified. Everyone else in the chat is stored, not classified.
    client_usernames: frozenset[str] = frozenset()
    # Who gets the reports in a private chat (after writing anything to the bot once). They can run commands.
    report_usernames: frozenset[str] = frozenset()
    # Contract for chats that have no explicit binding in the chats table.
    default_contract: str = "aag_tenders"

    def is_client(self, username: str | None) -> bool:
        return bool(username) and username.lower() in self.client_usernames

    def is_recipient(self, username: str | None) -> bool:
        return bool(username) and username.lower() in self.report_usernames

    @classmethod
    def from_env(cls) -> "Settings":
        env = os.environ.get
        return cls(
            telegram_token=env("TELEGRAM_BOT_TOKEN", ""),
            webhook_secret=env("TELEGRAM_WEBHOOK_SECRET", ""),
            supabase_url=env("SUPABASE_URL", "").rstrip("/"),
            supabase_key=env("SUPABASE_SERVICE_ROLE_KEY", ""),
            client_usernames=_names(env("CLIENT_USERNAMES", "")),
            report_usernames=_names(env("REPORT_TO", "")),
            default_contract=env("DEFAULT_CONTRACT", "aag_tenders"),
        )

    def missing(self) -> list[str]:
        required = {
            "TELEGRAM_BOT_TOKEN": self.telegram_token,
            "TELEGRAM_WEBHOOK_SECRET": self.webhook_secret,
            "SUPABASE_URL": self.supabase_url,
            "SUPABASE_SERVICE_ROLE_KEY": self.supabase_key,
            "CLIENT_USERNAMES": ",".join(self.client_usernames),
            "REPORT_TO": ",".join(self.report_usernames),
        }
        return [name for name, value in required.items() if not value]

"""Bot settings from environment variables (Vercel project env)."""

from __future__ import annotations

import os
from dataclasses import dataclass


def _ints(raw: str) -> frozenset[int]:
    return frozenset(int(x) for x in raw.replace(" ", "").split(",") if x)


def _names(raw: str) -> frozenset[str]:
    return frozenset(x.strip().lstrip("@").lower() for x in raw.split(",") if x.strip())


@dataclass(frozen=True)
class Settings:
    telegram_token: str
    # Telegram sends it in X-Telegram-Bot-Api-Secret-Token; requests without it are rejected.
    webhook_secret: str
    supabase_url: str
    supabase_key: str
    # Internal chat for reports. None = store only, never post.
    notify_chat_id: int | None = None
    # Our people: their messages are stored but not classified, and only they can run commands.
    team_user_ids: frozenset[int] = frozenset()
    team_usernames: frozenset[str] = frozenset()
    # Contract for chats that have no explicit binding in the chats table.
    default_contract: str = "aag_tenders"

    def is_team(self, user_id: int | None, username: str | None) -> bool:
        return (user_id in self.team_user_ids) or bool(username and username.lower() in self.team_usernames)

    @classmethod
    def from_env(cls) -> "Settings":
        env = os.environ.get
        notify = env("NOTIFY_CHAT_ID", "").strip()
        return cls(
            telegram_token=env("TELEGRAM_BOT_TOKEN", ""),
            webhook_secret=env("TELEGRAM_WEBHOOK_SECRET", ""),
            supabase_url=env("SUPABASE_URL", "").rstrip("/"),
            supabase_key=env("SUPABASE_SERVICE_ROLE_KEY", ""),
            notify_chat_id=int(notify) if notify else None,
            team_user_ids=_ints(env("TEAM_USER_IDS", "")),
            team_usernames=_names(env("TEAM_USERNAMES", "")),
            default_contract=env("DEFAULT_CONTRACT", "aag_tenders"),
        )

    def missing(self) -> list[str]:
        required = {
            "TELEGRAM_BOT_TOKEN": self.telegram_token,
            "TELEGRAM_WEBHOOK_SECRET": self.webhook_secret,
            "SUPABASE_URL": self.supabase_url,
            "SUPABASE_SERVICE_ROLE_KEY": self.supabase_key,
        }
        return [name for name, value in required.items() if not value]

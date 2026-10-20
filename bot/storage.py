"""Supabase via its REST API (PostgREST) with the service role key. Schema: supabase/migrations."""

from __future__ import annotations

import httpx


class StorageError(RuntimeError):
    pass


class Storage:
    def __init__(self, url: str, key: str, client: httpx.Client | None = None):
        self.base = f"{url}/rest/v1"
        self.client = client or httpx.Client(timeout=15)
        self.headers = {"apikey": key, "Authorization": f"Bearer {key}", "Content-Type": "application/json"}

    def _request(self, method: str, path: str, *, params: dict | None = None, json=None, prefer: str | None = None):
        headers = dict(self.headers)
        if prefer:
            headers["Prefer"] = prefer
        response = self.client.request(method, f"{self.base}/{path}", params=params, json=json, headers=headers)
        if response.is_error:
            raise StorageError(f"{method} {path}: HTTP {response.status_code} {response.text[:300]}")
        return response.json() if response.content else None

    def upsert_message(self, row: dict) -> None:
        self._request("POST", "messages", params={"on_conflict": "chat_id,message_id"}, json=row,
                      prefer="resolution=merge-duplicates,return=minimal")

    def get_message(self, chat_id: int, message_id: int) -> dict | None:
        rows = self._request("GET", "messages", params={
            "chat_id": f"eq.{chat_id}", "message_id": f"eq.{message_id}", "select": "text,category,error",
        })
        return rows[0] if rows else None

    def chat_contract(self, chat_id: int) -> str | None:
        rows = self._request("GET", "chats", params={"chat_id": f"eq.{chat_id}", "select": "contract_id"})
        return rows[0]["contract_id"] if rows else None

    def upsert_chat(self, chat_id: int, title: str, contract_id: str | None = None) -> None:
        row: dict = {"chat_id": chat_id, "title": title}
        if contract_id is not None:
            row["contract_id"] = contract_id
        self._request("POST", "chats", params={"on_conflict": "chat_id"}, json=row,
                      prefer="resolution=merge-duplicates,return=minimal")


class MemoryStorage:
    """Same interface as Storage, kept in process memory: for local runs and tests without Supabase."""

    def __init__(self):
        self.messages: dict[tuple[int, int], dict] = {}
        self.chats: dict[int, dict] = {}

    def upsert_message(self, row: dict) -> None:
        key = (row["chat_id"], row["message_id"])
        self.messages[key] = {**self.messages.get(key, {}), **row}

    def get_message(self, chat_id: int, message_id: int) -> dict | None:
        return self.messages.get((chat_id, message_id))

    def chat_contract(self, chat_id: int) -> str | None:
        return self.chats.get(chat_id, {}).get("contract_id")

    def upsert_chat(self, chat_id: int, title: str, contract_id: str | None = None) -> None:
        row = self.chats.setdefault(chat_id, {"title": title})
        if contract_id is not None:
            row["contract_id"] = contract_id

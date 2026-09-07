"""Cache API responses in SQLite to avoid repeating requests."""

import hashlib
import json
import sqlite3
from collections.abc import Callable
from contextlib import closing
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.config import DATA_DIR, require_secret

if TYPE_CHECKING:
    from anthropic import Anthropic
    from openai import OpenAI


class APICache:
    def __init__(
        self, api_call: Callable[..., Any], service: str, cache_path: str | Path | None = None
    ) -> None:
        """Set up a cache for API responses.

        api_call: Provider method called with the generation arguments.
        service: Provider and endpoint identifier used to separate cache entries.
        """
        self.api_call = api_call
        self.service = service
        self.cache_path = (
            Path(cache_path)
            if cache_path is not None
            else DATA_DIR / "cache" / "generations.sqlite3"
        )

    def generate(self, overwrite_cache: bool = False, **kwargs: Any) -> Any:
        """Return a cached response, or call the API and save the response.

        overwrite_cache: Make a new request even if a response is already cached.
        kwargs: Arguments passed to the provider method.
        """
        query = json.dumps(
            {"service": self.service, "request": kwargs},
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        key = hashlib.sha256(query.encode()).hexdigest()
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.cache_path)) as database, database:
            database.execute(
                "CREATE TABLE IF NOT EXISTS responses (key TEXT PRIMARY KEY, response TEXT NOT NULL)"
            )
            if not overwrite_cache:
                cached = database.execute(
                    "SELECT response FROM responses WHERE key = ?", (key,)
                ).fetchone()
                if cached is not None:
                    return json.loads(cached[0])
            response = self.api_call(**kwargs)
            if hasattr(response, "model_dump"):
                response = response.model_dump(mode="json")
            encoded = json.dumps(response, ensure_ascii=False, allow_nan=False)
            database.execute("INSERT OR REPLACE INTO responses VALUES (?, ?)", (key, encoded))
            return json.loads(encoded)


class OpenAIAPICache(APICache):
    def __init__(
        self,
        mode: str = "chat",
        cache_path: str | Path | None = None,
        client: "OpenAI | None" = None,
    ) -> None:
        if mode not in {"chat", "completion"}:
            raise ValueError("mode must be chat or completion")
        if client is None:
            from openai import OpenAI

            client = OpenAI(api_key=require_secret("OPENAI_API_KEY"), max_retries=2, timeout=120)
        create = client.chat.completions.create if mode == "chat" else client.completions.create
        super().__init__(create, f"openai/{mode}/{client.base_url}", cache_path)


class ClaudeAPICache(APICache):
    def __init__(
        self, client: "Anthropic | None" = None, cache_path: str | Path | None = None
    ) -> None:
        if client is None:
            from anthropic import Anthropic

            client = Anthropic(
                api_key=require_secret("ANTHROPIC_API_KEY"), max_retries=2, timeout=120
            )
        super().__init__(
            client.messages.create, f"anthropic/messages/{client.base_url}", cache_path
        )

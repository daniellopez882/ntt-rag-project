"""Test fixtures: a configured environment, stubbed database functions and a fake agent.

Nothing here reaches PostgreSQL or OpenAI. The API module imports the
database functions by name, so they are replaced on the module.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import pytest
from fastapi.testclient import TestClient

os.environ.update(
    {
        "API_KEY": "test-api-key",
        "OPENAI_API_KEY": "",
        "DB_PASSWORD": "test",
        "APP_ENV": "test",
        "CORS_ORIGINS": "http://ui.example",
    }
)
os.environ.pop("DATABASE_URL", None)

from agent import agent as agent_module  # noqa: E402
from agent import api  # noqa: E402
from agent.config import reload_settings  # noqa: E402

AUTH = {"X-API-Key": "test-api-key"}


@dataclass
class FakeDB:
    """In-memory stand-in for the session/message functions the API uses."""

    sessions: dict[str, dict[str, Any]] = field(default_factory=dict)
    messages: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    connected: bool = True
    counter: int = 0

    async def create_session(self, user_id=None, metadata=None, timeout_minutes=60):
        self.counter += 1
        session_id = f"00000000-0000-0000-0000-{self.counter:012d}"
        self.sessions[session_id] = {
            "id": session_id,
            "user_id": user_id,
            "metadata": metadata or {},
            "created_at": "2026-01-01T00:00:00",
            "updated_at": "2026-01-01T00:00:00",
            "expires_at": None,
            "timeout_minutes": timeout_minutes,
        }
        self.messages[session_id] = []
        return session_id

    async def get_session(self, session_id):
        return self.sessions.get(session_id)

    async def add_message(self, session_id, role, content, metadata=None):
        self.messages.setdefault(session_id, []).append(
            {"role": role, "content": content, "metadata": metadata or {}}
        )
        return f"m{len(self.messages[session_id])}"

    async def get_session_messages(self, session_id, limit=None):
        msgs = self.messages.get(session_id, [])
        return msgs[:limit] if limit else msgs

    async def test_connection(self):
        return self.connected


class FakePart:
    def __init__(self, part_kind, **attrs):
        self.part_kind = part_kind
        for k, v in attrs.items():
            setattr(self, k, v)


class FakeMessage:
    def __init__(self, parts):
        self.parts = parts


class FakeResult:
    def __init__(self, output, messages=None):
        self.output = output
        self._messages = messages or []

    def all_messages(self):
        return self._messages


class FakeAgent:
    """Records prompts; returns a canned result or raises."""

    def __init__(
        self, output="Hello from the agent", messages=None, error: Exception | None = None
    ):
        self.output = output
        self.messages = messages or []
        self.error = error
        self.prompts: list[str] = []
        self.deps: list[Any] = []

    async def run(self, prompt, deps=None):
        self.prompts.append(prompt)
        self.deps.append(deps)
        if self.error:
            raise self.error
        return FakeResult(self.output, self.messages)


@pytest.fixture(autouse=True)
def _settings(monkeypatch):
    reload_settings()
    yield
    reload_settings()


@pytest.fixture
def db(monkeypatch) -> FakeDB:
    fake = FakeDB()
    for name in (
        "create_session",
        "get_session",
        "add_message",
        "get_session_messages",
        "test_connection",
    ):
        monkeypatch.setattr(api, name, getattr(fake, name))
    return fake


@pytest.fixture
def fake_agent() -> FakeAgent:
    agent = FakeAgent()
    agent_module.set_agent(agent)  # type: ignore[arg-type]
    yield agent
    agent_module.set_agent(None)


@pytest.fixture
def client(db, fake_agent) -> TestClient:
    # No `with`: the lifespan (real database init) is not run.
    return TestClient(api.app, raise_server_exceptions=False)

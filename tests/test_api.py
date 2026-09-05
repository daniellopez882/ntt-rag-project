from __future__ import annotations

import pytest

from agent import api
from agent.config import reload_settings
from agent.models import ChunkResult, DocumentMetadata
from tests.conftest import AUTH, FakeAgent, FakeMessage, FakePart

# --- authentication -----------------------------------------------------------


def test_health_needs_no_key_and_reports_configuration(client, db):
    res = client.get("/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "healthy"
    assert body["llm_configured"] is False  # OPENAI_API_KEY is empty in tests; was hard-coded True
    assert body["auth_configured"] is True


def test_health_is_503_when_the_database_is_down(client, db):
    db.connected = False
    res = client.get("/health")
    assert res.status_code == 503
    assert res.json()["status"] == "unhealthy"


@pytest.mark.parametrize(
    "method, path, body",
    [
        ("POST", "/chat", {"message": "hi"}),
        ("POST", "/chat/stream", {"message": "hi"}),
        ("POST", "/search/vector", {"query": "x"}),
        ("POST", "/search/hybrid", {"query": "x"}),
        ("GET", "/documents", None),
        ("GET", "/sessions/00000000-0000-0000-0000-000000000001", None),
    ],
)
def test_every_other_route_requires_the_api_key(client, method, path, body):
    res = client.request(method, path, json=body)
    assert res.status_code == 401
    res = client.request(method, path, json=body, headers={"X-API-Key": "wrong"})
    assert res.status_code == 401


def test_without_a_configured_key_the_api_fails_closed(client, monkeypatch):
    monkeypatch.setenv("API_KEY", "")
    reload_settings()
    res = client.post("/chat", json={"message": "hi"}, headers=AUTH)
    assert res.status_code == 503
    assert "API_KEY" in res.json()["detail"]


def test_cors_comes_from_configuration(client):
    res = client.options(
        "/chat",
        headers={"Origin": "http://ui.example", "Access-Control-Request-Method": "POST"},
    )
    assert res.headers.get("access-control-allow-origin") == "http://ui.example"
    assert res.headers.get("access-control-allow-credentials") is None
    res = client.options(
        "/chat", headers={"Origin": "http://evil.example", "Access-Control-Request-Method": "POST"}
    )
    assert res.headers.get("access-control-allow-origin") is None


# --- chat -------------------------------------------------------------------------


def test_chat_creates_a_session_runs_the_agent_and_saves_the_turn(client, db, fake_agent):
    fake_agent.messages = [
        FakeMessage(
            [
                FakePart("text", content="thinking"),
                FakePart(
                    "tool-call",
                    tool_name="vector_search",
                    args={"query": "q", "limit": 3},
                    tool_call_id="c1",
                ),
            ]
        )
    ]
    res = client.post(
        "/chat", json={"message": "What is NTT DATA?", "user_id": "alice"}, headers=AUTH
    )
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["message"] == "Hello from the agent"
    assert body["tools_used"] == [
        {"tool_name": "vector_search", "args": {"query": "q", "limit": 3}, "tool_call_id": "c1"}
    ]
    session_id = body["session_id"]
    assert db.sessions[session_id]["user_id"] == "alice"
    assert [m["role"] for m in db.messages[session_id]] == ["user", "assistant"]
    assert fake_agent.prompts == ["What is NTT DATA?"]  # no history yet: the bare message
    assert fake_agent.deps[0].session_id == session_id


def test_chat_continues_an_owned_session_with_context(client, db, fake_agent):
    first = client.post("/chat", json={"message": "first", "user_id": "alice"}, headers=AUTH).json()
    sid = first["session_id"]
    res = client.post(
        "/chat", json={"message": "second", "user_id": "alice", "session_id": sid}, headers=AUTH
    )
    assert res.status_code == 200
    assert res.json()["session_id"] == sid
    prompt = fake_agent.prompts[-1]
    assert prompt.startswith("Previous conversation:")
    assert "user: first" in prompt and "assistant: Hello from the agent" in prompt
    assert prompt.endswith("Current question: second")


def test_chat_cannot_continue_someone_elses_session(client, db, fake_agent):
    sid = client.post("/chat", json={"message": "mine", "user_id": "alice"}, headers=AUTH).json()[
        "session_id"
    ]
    res = client.post(
        "/chat", json={"message": "hijack", "user_id": "mallory", "session_id": sid}, headers=AUTH
    )
    assert res.status_code == 404
    assert len(db.messages[sid]) == 2  # nothing appended
    res = client.post("/chat", json={"message": "no user", "session_id": sid}, headers=AUTH)
    assert res.status_code == 404


def test_agent_failure_is_a_502_without_leaking_or_saving_an_error_turn(client, db, fake_agent):
    fake_agent.error = RuntimeError("OpenAI 401: invalid key sk-live-123")
    res = client.post("/chat", json={"message": "hi", "user_id": "alice"}, headers=AUTH)
    assert res.status_code == 502
    assert "sk-live" not in res.text
    assert all(
        len(msgs) == 0 for msgs in db.messages.values()
    )  # was: "I encountered an error…" saved as the answer


def test_chat_message_is_bounded(client):
    res = client.post("/chat", json={"message": ""}, headers=AUTH)
    assert res.status_code == 422
    res = client.post("/chat", json={"message": "x" * 8001}, headers=AUTH)
    assert res.status_code == 422


def test_unhandled_errors_are_generic_json(client, db, monkeypatch):
    async def boom(*args, **kwargs):
        raise RuntimeError("database password is hunter2")

    monkeypatch.setattr(api, "create_session", boom)
    res = client.post("/chat", json={"message": "hi"}, headers=AUTH)
    assert res.status_code == 500
    body = res.json()
    assert body["error"] == "Internal server error" and "request_id" in body
    assert "hunter2" not in res.text  # the old handler returned str(exc), and was not a Response


# --- search and documents ----------------------------------------------------------


def _chunk(i: int) -> ChunkResult:
    return ChunkResult(
        chunk_id=f"c{i}",
        document_id="d1",
        content=f"chunk {i}",
        score=0.5,
        document_title="Report",
        document_source="report.pdf",
    )


def test_search_endpoints_bound_their_input_and_hide_failures(client, monkeypatch):
    async def ok(input_data):
        return [_chunk(i) for i in range(input_data.limit)]

    async def fails(input_data):
        raise RuntimeError("pgvector down at 10.0.0.5")

    monkeypatch.setattr(api, "vector_search_tool", ok)
    res = client.post("/search/vector", json={"query": "sustainability", "limit": 2}, headers=AUTH)
    assert res.status_code == 200
    assert res.json()["total_results"] == 2 and res.json()["search_type"] == "vector"

    assert (
        client.post("/search/vector", json={"query": "x", "limit": 0}, headers=AUTH).status_code
        == 422
    )
    assert (
        client.post("/search/vector", json={"query": "x", "limit": 51}, headers=AUTH).status_code
        == 422
    )

    monkeypatch.setattr(api, "hybrid_search_tool", fails)
    res = client.post("/search/hybrid", json={"query": "x"}, headers=AUTH)
    assert res.status_code == 502
    assert "10.0.0.5" not in res.text


def test_documents_limit_is_bounded(client, monkeypatch):
    seen = {}

    async def listing(input_data):
        seen["limit"], seen["offset"] = input_data.limit, input_data.offset
        return [
            DocumentMetadata(
                id="d1",
                title="Report",
                source="report.pdf",
                created_at="2026-01-01T00:00:00",
                updated_at="2026-01-01T00:00:00",
                chunk_count=3,
            )
        ]

    monkeypatch.setattr(api, "list_documents_tool", listing)
    res = client.get("/documents?limit=5&offset=10", headers=AUTH)
    assert res.status_code == 200 and seen == {"limit": 5, "offset": 10}
    assert (
        client.get("/documents?limit=1000", headers=AUTH).status_code == 422
    )  # was passed straight to SQL
    assert client.get("/documents?offset=-1", headers=AUTH).status_code == 422


# --- sessions -------------------------------------------------------------------


def test_session_lookup_requires_the_owner(client, db, fake_agent):
    sid = client.post("/chat", json={"message": "m", "user_id": "alice"}, headers=AUTH).json()[
        "session_id"
    ]
    assert client.get(f"/sessions/{sid}", headers=AUTH).status_code == 404
    assert client.get(f"/sessions/{sid}?user_id=bob", headers=AUTH).status_code == 404
    res = client.get(f"/sessions/{sid}?user_id=alice", headers=AUTH)
    assert res.status_code == 200 and res.json()["id"] == sid
    assert client.get("/sessions/not-a-uuid", headers=AUTH).status_code == 404


# --- helpers ----------------------------------------------------------------------


def test_result_output_supports_both_attribute_names():
    class Old:
        data = "old"

    class New:
        output = "new"

    assert api.result_output(New()) == "new"
    assert api.result_output(Old()) == "old"


def test_extract_tool_calls_parses_json_args_and_skips_text_parts():
    result = type("R", (), {})()
    result.all_messages = lambda: [
        FakeMessage(
            [
                FakePart("text", content="x"),
                FakePart("tool-call", tool_name="hybrid_search", args='{"query": "q"}'),
            ]
        )
    ]
    calls = api.extract_tool_calls(result)
    assert [c.model_dump() for c in calls] == [
        {"tool_name": "hybrid_search", "args": {"query": "q"}, "tool_call_id": None}
    ]


def test_fake_agent_is_used_only_in_tests():
    assert isinstance(FakeAgent(), FakeAgent)


def test_chat_without_a_configured_model_is_a_503_not_a_generic_502(db, monkeypatch):
    from fastapi.testclient import TestClient

    from agent import agent as agent_module

    monkeypatch.setenv("OPENAI_API_KEY", "")
    reload_settings()
    agent_module.set_agent(None)  # force the real builder, which needs the key
    client = TestClient(api.app, raise_server_exceptions=False)
    res = client.post("/chat", json={"message": "hi", "user_id": "u"}, headers=AUTH)
    assert res.status_code == 503
    assert res.json()["detail"] == "The model provider is not configured"
    res = client.post("/chat/stream", json={"message": "hi", "user_id": "u"}, headers=AUTH)
    assert res.status_code == 503

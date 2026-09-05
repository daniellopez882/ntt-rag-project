from __future__ import annotations

import pytest
from pydantic import ValidationError

from agent.config import ConfigurationError, Settings, reload_settings
from agent.providers import get_llm_model, reset_clients
from agent.tools import DocumentInput, DocumentListInput, HybridSearchInput, VectorSearchInput


def test_settings_defaults_and_database_url(monkeypatch):
    for k in ("API_KEY", "CORS_ORIGINS", "DATABASE_URL", "OPENAI_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    s = Settings(_env_file=None, DB_PASSWORD="pw")
    assert s.database_url == "postgresql://postgres:pw@postgres:5432/vector_db"
    assert s.cors_origins == [] and s.auth_configured is False and s.llm_configured is False
    s = Settings(
        _env_file=None, DATABASE_URL="postgresql://u:p@h:1/db", CORS_ORIGINS="http://a, http://b ,"
    )
    assert s.database_url == "postgresql://u:p@h:1/db"
    assert s.cors_origins == ["http://a", "http://b"]


def test_settings_bounds():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, APP_PORT=0)
    with pytest.raises(ValidationError):
        Settings(_env_file=None, MAX_CONTEXT_MESSAGES=101)


def test_missing_openai_key_is_a_configuration_error(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "")
    reload_settings()
    reset_clients()
    with pytest.raises(ConfigurationError, match="OPENAI_API_KEY"):
        get_llm_model()


def test_search_inputs_are_bounded():
    assert VectorSearchInput(query="q").limit == 10
    with pytest.raises(ValidationError):
        VectorSearchInput(query="q", limit=0)
    with pytest.raises(ValidationError):
        VectorSearchInput(query="q", limit=51)
    with pytest.raises(ValidationError):
        VectorSearchInput(query="")
    with pytest.raises(ValidationError):
        HybridSearchInput(query="q", text_weight=1.5)
    with pytest.raises(ValidationError):
        DocumentListInput(limit=101)
    with pytest.raises(ValidationError):
        DocumentListInput(offset=-1)


def test_document_id_must_be_a_uuid():
    ok = DocumentInput(document_id="123E4567-E89B-12D3-A456-426614174000")
    assert ok.document_id == "123e4567-e89b-12d3-a456-426614174000"
    with pytest.raises(ValidationError):
        DocumentInput(document_id="1 OR 1=1")

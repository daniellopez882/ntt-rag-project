"""Configuration.

Every value the modules read with ad-hoc ``os.getenv`` calls — database
parts, the OpenAI key (which defaulted to the literal string ``"LLM_API_KEY"``),
model names, ports — is a setting here, plus the two things the API lacked:
an API key for its callers and an explicit CORS origin list.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ConfigurationError(RuntimeError):
    """A required setting is missing."""


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    APP_ENV: str = "development"
    LOG_LEVEL: str = "INFO"
    APP_HOST: str = "0.0.0.0"  # noqa: S104  # nosec B104 - bound inside a container
    APP_PORT: int = Field(default=8058, ge=1, le=65535)

    # Callers must send this in X-API-Key. Empty means every route but /health
    # answers 503 until it is set: the API fails closed, not open.
    API_KEY: str = ""
    # Comma-separated browser origins allowed to call the API. Empty = none.
    CORS_ORIGINS: str = ""

    DB_USER: str = "postgres"
    DB_PASSWORD: str = "postgres"
    DB_HOST: str = "postgres"
    DB_PORT: int = 5432
    DB_NAME: str = "vector_db"
    DATABASE_URL: str = ""  # overrides the parts above when set

    OPENAI_API_KEY: str = ""
    LLM_CHOICE: str = "gpt-4o-mini"
    EMBEDDING_MODEL: str = "text-embedding-3-small"

    MAX_MESSAGE_CHARS: int = Field(default=8000, ge=1)
    MAX_CONTEXT_MESSAGES: int = Field(default=10, ge=0, le=100)
    SESSION_TIMEOUT_MINUTES: int = Field(default=60, ge=1)

    @property
    def database_url(self) -> str:
        if self.DATABASE_URL:
            return self.DATABASE_URL
        return (
            f"postgresql://{self.DB_USER}:{self.DB_PASSWORD}"
            f"@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        )

    @property
    def cors_origins(self) -> list[str]:
        return [o.strip() for o in self.CORS_ORIGINS.split(",") if o.strip()]

    @property
    def is_production(self) -> bool:
        return self.APP_ENV.lower() == "production"

    @property
    def llm_configured(self) -> bool:
        return bool(self.OPENAI_API_KEY.strip())

    @property
    def auth_configured(self) -> bool:
        return bool(self.API_KEY.strip())


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def reload_settings() -> Settings:
    """Rebuild from the current environment (tests)."""
    get_settings.cache_clear()
    return get_settings()

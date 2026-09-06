"""Model and embedding providers, built on demand.

The previous module read ``OPENAI_API_KEY`` with a default of the literal
string ``"LLM_API_KEY"`` — a typo for an environment variable name — so a
missing key produced an authentication failure from OpenAI at request time
instead of a clear configuration error. Clients are constructed lazily and
only when a key exists.
"""

from __future__ import annotations

from functools import lru_cache

import openai
from pydantic_ai.providers.openai import OpenAIProvider

from .config import ConfigurationError, get_settings

try:  # pydantic-ai renamed the class; support both spellings.
    from pydantic_ai.models.openai import OpenAIChatModel
except ImportError:  # pragma: no cover - depends on the installed version
    from pydantic_ai.models.openai import OpenAIModel as OpenAIChatModel  # type: ignore[assignment]


def _api_key() -> str:
    key = get_settings().OPENAI_API_KEY.strip()
    if not key:
        raise ConfigurationError("OPENAI_API_KEY is not set")
    return key


def get_llm_model(model_choice: str | None = None) -> OpenAIChatModel:
    """The chat model the agent runs on."""
    return OpenAIChatModel(
        model_choice or get_settings().LLM_CHOICE, provider=OpenAIProvider(api_key=_api_key())
    )


@lru_cache(maxsize=1)
def get_embedding_client() -> openai.AsyncOpenAI:
    """A shared async client for embeddings; created on first use, not at import."""
    return openai.AsyncOpenAI(api_key=_api_key())


def get_embedding_model() -> str:
    return get_settings().EMBEDDING_MODEL


def reset_clients() -> None:
    """Tests: drop the cached client."""
    get_embedding_client.cache_clear()

"""The Pydantic AI agent and its tools.

Built on first use: the previous module created ``Agent(get_llm_model(), …)``
at import, so importing ``agent.api`` — or running a test — needed a working
OpenAI key. The tool wrappers catch failures and return empty results so the
model can say so; the underlying tools (``tools.py``) raise.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from pydantic_ai import Agent, RunContext

from .prompts import SYSTEM_PROMPT
from .providers import get_llm_model
from .tools import (
    DocumentInput,
    DocumentListInput,
    HybridSearchInput,
    VectorSearchInput,
    get_document_tool,
    hybrid_search_tool,
    list_documents_tool,
    vector_search_tool,
)

logger = logging.getLogger(__name__)


@dataclass
class AgentDependencies:
    session_id: str
    user_id: str | None = None
    search_preferences: dict[str, Any] = field(
        default_factory=lambda: {"use_vector": True, "default_limit": 10}
    )


async def vector_search(
    ctx: RunContext[AgentDependencies], query: str, limit: int = 10
) -> list[dict[str, Any]]:
    """Search for relevant information using semantic similarity.

    Args:
        query: Search query to find similar content
        limit: Maximum number of results to return (1-50)
    """
    try:
        results = await vector_search_tool(VectorSearchInput(query=query, limit=limit))
    except Exception:
        logger.exception("vector_search failed")
        return []
    return [
        r.model_dump(include={"content", "score", "document_title", "document_source", "chunk_id"})
        for r in results
    ]


async def hybrid_search(
    ctx: RunContext[AgentDependencies], query: str, limit: int = 10, text_weight: float = 0.3
) -> list[dict[str, Any]]:
    """Combine semantic similarity with keyword matching.

    Args:
        query: Search query for hybrid search
        limit: Maximum number of results to return (1-50)
        text_weight: Weight for text similarity vs vector similarity (0.0-1.0)
    """
    try:
        results = await hybrid_search_tool(
            HybridSearchInput(query=query, limit=limit, text_weight=text_weight)
        )
    except Exception:
        logger.exception("hybrid_search failed")
        return []
    return [
        r.model_dump(include={"content", "score", "document_title", "document_source", "chunk_id"})
        for r in results
    ]


async def get_document(
    ctx: RunContext[AgentDependencies], document_id: str
) -> dict[str, Any] | None:
    """Retrieve the complete content of a specific document.

    Args:
        document_id: UUID of the document to retrieve
    """
    try:
        document = await get_document_tool(DocumentInput(document_id=document_id))
    except Exception:
        logger.exception("get_document failed")
        return None
    if not document:
        return None
    return {
        "id": document["id"],
        "title": document["title"],
        "source": document["source"],
        "content": document["content"],
        "chunk_count": len(document.get("chunks", [])),
        "created_at": document["created_at"],
    }


async def list_documents(
    ctx: RunContext[AgentDependencies], limit: int = 20, offset: int = 0
) -> list[dict[str, Any]]:
    """List available documents with their metadata.

    Args:
        limit: Maximum number of documents to return (1-100)
        offset: Number of documents to skip for pagination
    """
    try:
        documents = await list_documents_tool(DocumentListInput(limit=limit, offset=offset))
    except Exception:
        logger.exception("list_documents failed")
        return []
    return [
        {
            "id": d.id,
            "title": d.title,
            "source": d.source,
            "chunk_count": d.chunk_count,
            "created_at": d.created_at.isoformat(),
        }
        for d in documents
    ]


def build_agent(model: Any = None) -> Agent[AgentDependencies, str]:
    """Construct the agent; ``model`` lets tests pass a TestModel."""
    agent: Agent[AgentDependencies, str] = Agent(
        model or get_llm_model(), deps_type=AgentDependencies, system_prompt=SYSTEM_PROMPT
    )
    for tool in (vector_search, hybrid_search, get_document, list_documents):
        agent.tool(tool)
    return agent


_agent: Agent[AgentDependencies, str] | None = None


def get_agent() -> Agent[AgentDependencies, str]:
    global _agent
    if _agent is None:
        _agent = build_agent()
    return _agent


def set_agent(agent: Agent[AgentDependencies, str] | None) -> None:
    """Tests: replace the process-wide agent."""
    global _agent
    _agent = agent

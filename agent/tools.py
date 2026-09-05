"""Search and document tools, shared by the agent and the API.

Inputs are bounded (the old models accepted any ``limit``; ``/documents``
would pass 10**9 straight to ``LIMIT``), document ids must be UUIDs before
they reach ``::uuid`` casts, and the embedding client is created on first
use — the old module built it at import, so nothing here could be imported
without an OpenAI key. These functions raise on failure; the agent-facing
wrappers in ``agent.py`` decide what the model sees.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field, field_validator

from .db_utils import (
    get_document,
    get_document_chunks,
    hybrid_search,
    list_documents,
    vector_search,
)
from .models import ChunkResult, DocumentMetadata
from .providers import get_embedding_client, get_embedding_model

logger = logging.getLogger(__name__)

MAX_SEARCH_LIMIT = 50
MAX_LIST_LIMIT = 100


async def generate_embedding(text: str) -> list[float]:
    response = await get_embedding_client().embeddings.create(
        model=get_embedding_model(), input=text
    )
    return response.data[0].embedding


class VectorSearchInput(BaseModel):
    query: str = Field(..., min_length=1, max_length=4000, description="Search query")
    limit: int = Field(default=10, ge=1, le=MAX_SEARCH_LIMIT)


class HybridSearchInput(VectorSearchInput):
    text_weight: float = Field(default=0.3, ge=0.0, le=1.0)


class DocumentInput(BaseModel):
    document_id: str = Field(..., description="Document UUID")

    @field_validator("document_id")
    @classmethod
    def _must_be_uuid(cls, value: str) -> str:
        return str(UUID(value))


class DocumentListInput(BaseModel):
    limit: int = Field(default=20, ge=1, le=MAX_LIST_LIMIT)
    offset: int = Field(default=0, ge=0)


def _chunk(row: dict[str, Any], score_key: str) -> ChunkResult:
    return ChunkResult(
        chunk_id=str(row["chunk_id"]),
        document_id=str(row["document_id"]),
        content=row["content"],
        score=row[score_key],
        metadata=row["metadata"],
        document_title=row["document_title"],
        document_source=row["document_source"],
    )


async def vector_search_tool(input_data: VectorSearchInput) -> list[ChunkResult]:
    embedding = await generate_embedding(input_data.query)
    rows = await vector_search(embedding=embedding, limit=input_data.limit)
    return [_chunk(r, "similarity") for r in rows]


async def hybrid_search_tool(input_data: HybridSearchInput) -> list[ChunkResult]:
    embedding = await generate_embedding(input_data.query)
    rows = await hybrid_search(
        embedding=embedding,
        query_text=input_data.query,
        limit=input_data.limit,
        text_weight=input_data.text_weight,
    )
    return [_chunk(r, "combined_score") for r in rows]


async def get_document_tool(input_data: DocumentInput) -> dict[str, Any] | None:
    document = await get_document(input_data.document_id)
    if document:
        document["chunks"] = await get_document_chunks(input_data.document_id)
    return document


async def list_documents_tool(input_data: DocumentListInput) -> list[DocumentMetadata]:
    rows = await list_documents(limit=input_data.limit, offset=input_data.offset)
    return [
        DocumentMetadata(
            id=d["id"],
            title=d["title"],
            source=d["source"],
            metadata=d["metadata"],
            created_at=datetime.fromisoformat(d["created_at"]),
            updated_at=datetime.fromisoformat(d["updated_at"]),
            chunk_count=d.get("chunk_count"),
        )
        for d in rows
    ]

from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class SearchType(StrEnum):
    """Search type enumeration."""

    VECTOR = "vector"
    HYBRID = "hybrid"


class ChunkResult(BaseModel):
    """Chunk search result model."""

    chunk_id: str
    document_id: str
    content: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)
    document_title: str
    document_source: str

    @field_validator("score")
    @classmethod
    def validate_score(cls, v: float) -> float:
        """Ensure score is between 0 and 1."""
        return max(0.0, min(1.0, v))


class SearchResponse(BaseModel):
    """Search response model."""

    results: list[ChunkResult] = Field(default_factory=list)
    total_results: int = 0
    search_type: SearchType
    query_time_ms: float


# Request Models
class ChatRequest(BaseModel):
    """Chat request model."""

    message: str = Field(..., min_length=1, max_length=8000, description="User message")
    session_id: str | None = Field(None, description="Session ID for conversation continuity")
    user_id: str | None = Field(None, description="User identifier")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    search_type: SearchType = Field(
        default=SearchType.HYBRID, description="Type of search to perform"
    )
    model_config = ConfigDict(use_enum_values=True)


class SearchRequest(BaseModel):
    """Search request model."""

    query: str = Field(..., description="Search query")
    search_type: SearchType = Field(default=SearchType.HYBRID, description="Type of search")
    limit: int = Field(default=10, ge=1, le=50, description="Maximum results")
    filters: dict[str, Any] = Field(default_factory=dict, description="Search filters")
    model_config = ConfigDict(use_enum_values=True)


# Response Models
class DocumentMetadata(BaseModel):
    """Document metadata model."""

    id: str
    title: str
    source: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime
    chunk_count: int | None = None


class ToolCall(BaseModel):
    """Tool call information model."""

    tool_name: str
    args: dict[str, Any] = Field(default_factory=dict)
    tool_call_id: str | None = None


class ChatResponse(BaseModel):
    """Chat response model."""

    message: str
    session_id: str
    sources: list[DocumentMetadata] = Field(default_factory=list)
    tools_used: list[ToolCall] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


# Ingestion Models
class IngestionConfig(BaseModel):
    """Configuration for document ingestion."""

    chunk_size: int = Field(default=850, ge=100, le=5000)
    chunk_overlap: int = Field(default=150, ge=0, le=1000)
    max_chunk_size: int = Field(default=2000, ge=500, le=10000)
    use_semantic_chunking: bool = True

    @field_validator("chunk_overlap")
    @classmethod
    def validate_overlap(cls, v: int, info) -> int:
        """Ensure overlap is less than chunk size."""
        chunk_size = info.data.get("chunk_size", 1000)
        if v >= chunk_size:
            raise ValueError(f"Chunk overlap ({v}) must be less than chunk size ({chunk_size})")
        return v


class IngestionResult(BaseModel):
    """Result of document ingestion."""

    document_id: str
    title: str
    chunks_created: int
    processing_time_ms: float


# Error Models
class ErrorResponse(BaseModel):
    """Error response model."""

    error: str
    error_type: str
    details: dict[str, Any] | None = None
    request_id: str | None = None


# Health Check Models
class HealthStatus(BaseModel):
    """Health check status."""

    status: Literal["healthy", "unhealthy"]
    database: bool
    llm_configured: bool  # was `llm_connection`, hard-coded True
    auth_configured: bool
    version: str
    timestamp: datetime

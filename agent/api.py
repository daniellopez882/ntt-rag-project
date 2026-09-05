"""The HTTP API.

What changed, each reproduced before the change:

* No route authenticated. ``X-API-Key`` is required everywhere but
  ``/health``; without a configured key the API answers 503 (closed), not 200.
* ``allow_origins=["*"]`` with ``allow_credentials=True`` — origins come from
  ``CORS_ORIGINS`` and credentials are off.
* Every error path returned ``str(e)`` to the caller, and the global handler
  returned a Pydantic model where Starlette needs a ``Response`` (a second
  crash on top of the first). Errors are generic JSON with a request id; the
  detail goes to the log.
* ``execute_agent`` caught every exception and saved
  ``"I encountered an error… <exception>"`` as the assistant's turn. Failures
  are failures now.
* Any caller could continue or read any session by id. A session is bound to
  the ``user_id`` declared when it was created.
* ``/documents`` passed unbounded ``limit``/``offset`` to SQL.
* ``result.data`` is ``result.output`` in the pinned pydantic-ai.
* ``HealthStatus.llm_connection`` was hard-coded ``True``.
"""

from __future__ import annotations

import json
import logging
import secrets
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import APIKeyHeader
from pydantic_ai.messages import PartDeltaEvent, PartStartEvent, TextPartDelta

from .agent import AgentDependencies, get_agent
from .config import get_settings
from .db_utils import (
    add_message,
    close_database,
    create_session,
    execute_init_sql,
    get_session,
    get_session_messages,
    initialize_database,
    test_connection,
)
from .models import (
    ChatRequest,
    ChatResponse,
    HealthStatus,
    SearchRequest,
    SearchResponse,
    SearchType,
    ToolCall,
)
from .tools import (
    DocumentListInput,
    HybridSearchInput,
    VectorSearchInput,
    hybrid_search_tool,
    list_documents_tool,
    vector_search_tool,
)

logger = logging.getLogger(__name__)

API_VERSION = "1.2.0"


class AgentError(RuntimeError):
    """The agent could not produce a response."""


# --- auth ---------------------------------------------------------------------

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def require_api_key(api_key: str | None = Depends(_api_key_header)) -> None:
    settings = get_settings()
    if not settings.auth_configured:
        raise HTTPException(status_code=503, detail="API_KEY is not configured on this server")
    if not api_key or not secrets.compare_digest(api_key, settings.API_KEY):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


# --- app --------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    logging.basicConfig(
        level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    if not settings.auth_configured:
        logger.warning("API_KEY is not set: every route except /health will answer 503")
    await initialize_database()
    await execute_init_sql("sql/schema.sql")
    if not await test_connection():
        logger.error("Database connection failed")
    logger.info("Startup complete")
    yield
    await close_database()


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="Agentic RAG",
        description="A Pydantic AI agent over pgvector search",
        version=API_VERSION,
        lifespan=lifespan,
    )
    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_credentials=False,
            allow_methods=["GET", "POST"],
            allow_headers=["Content-Type", "X-API-Key"],
        )
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    _register_routes(app)
    return app


# --- helpers ------------------------------------------------------------------


async def get_or_create_session(request: ChatRequest) -> str:
    """The caller's session: an existing one they own, or a new one."""
    if request.session_id:
        session = await get_session(request.session_id)
        if session and session.get("user_id") == request.user_id:
            return request.session_id
        raise HTTPException(status_code=404, detail="Session not found")
    return await create_session(
        user_id=request.user_id,
        metadata=request.metadata,
        timeout_minutes=get_settings().SESSION_TIMEOUT_MINUTES,
    )


async def build_prompt(session_id: str, message: str) -> str:
    """The message with the last few turns in front of it."""
    limit = get_settings().MAX_CONTEXT_MESSAGES
    if limit == 0:
        return message
    history = await get_session_messages(session_id, limit=limit)
    if not history:
        return message
    context = "\n".join(f"{m['role']}: {m['content']}" for m in history[-6:])
    return f"Previous conversation:\n{context}\n\nCurrent question: {message}"


def result_output(result: Any) -> str:
    """``AgentRunResult.output`` (``.data`` before pydantic-ai 0.4)."""
    if hasattr(result, "output"):
        return str(result.output)
    return str(result.data)


def extract_tool_calls(result: Any) -> list[ToolCall]:
    calls: list[ToolCall] = []
    try:
        messages = result.all_messages()
    except Exception:
        return calls
    for message in messages:
        for part in getattr(message, "parts", []):
            if getattr(part, "part_kind", None) != "tool-call":
                continue
            args: Any = getattr(part, "args", {})
            if hasattr(part, "args_as_dict"):
                try:
                    args = part.args_as_dict()
                except Exception:
                    args = {}
            elif isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            calls.append(
                ToolCall(
                    tool_name=str(getattr(part, "tool_name", "unknown")),
                    args=args if isinstance(args, dict) else {},
                    tool_call_id=getattr(part, "tool_call_id", None),
                )
            )
    return calls


async def execute_agent(
    message: str, session_id: str, user_id: str | None
) -> tuple[str, list[ToolCall]]:
    """Run the agent and persist the turn. Raises ``AgentError`` on failure."""
    deps = AgentDependencies(session_id=session_id, user_id=user_id)
    prompt = await build_prompt(session_id, message)
    try:
        result = await get_agent().run(prompt, deps=deps)
    except Exception as exc:
        logger.exception("Agent run failed")
        raise AgentError("The agent could not process the request") from exc
    response = result_output(result)
    tools_used = extract_tool_calls(result)
    await add_message(session_id, "user", message, {"user_id": user_id})
    await add_message(session_id, "assistant", response, {"tool_calls": len(tools_used)})
    return response, tools_used


def sse(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload)}\n\n"


# --- routes -------------------------------------------------------------------


def _register_routes(app: FastAPI) -> None:
    protected = [Depends(require_api_key)]

    @app.get("/health", response_model=HealthStatus)
    async def health_check() -> Any:
        settings = get_settings()
        db_ok = await test_connection()
        status = HealthStatus(
            status="healthy" if db_ok else "unhealthy",
            database=db_ok,
            llm_configured=settings.llm_configured,
            auth_configured=settings.auth_configured,
            version=API_VERSION,
            timestamp=datetime.now(),
        )
        if not db_ok:
            return JSONResponse(status_code=503, content=status.model_dump(mode="json"))
        return status

    @app.post("/chat", response_model=ChatResponse, dependencies=protected)
    async def chat(request: ChatRequest) -> ChatResponse:
        session_id = await get_or_create_session(request)
        try:
            response, tools_used = await execute_agent(request.message, session_id, request.user_id)
        except AgentError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        return ChatResponse(
            message=response,
            session_id=session_id,
            tools_used=tools_used,
            metadata={"search_type": str(request.search_type)},
        )

    @app.post("/chat/stream", dependencies=protected)
    async def chat_stream(request: ChatRequest) -> StreamingResponse:
        session_id = await get_or_create_session(request)
        prompt = await build_prompt(session_id, request.message)
        deps = AgentDependencies(session_id=session_id, user_id=request.user_id)
        agent = get_agent()

        async def generate() -> AsyncIterator[str]:
            yield sse({"type": "session", "session_id": session_id})
            full_response = ""
            try:
                await add_message(session_id, "user", request.message, {"user_id": request.user_id})
                async with agent.iter(prompt, deps=deps) as run:
                    async for node in run:
                        if not agent.is_model_request_node(node):
                            continue
                        async with node.stream(run.ctx) as stream:
                            async for event in stream:
                                delta = None
                                if (
                                    isinstance(event, PartStartEvent)
                                    and event.part.part_kind == "text"
                                ):
                                    delta = event.part.content
                                elif isinstance(event, PartDeltaEvent) and isinstance(
                                    event.delta, TextPartDelta
                                ):
                                    delta = event.delta.content_delta
                                if delta:
                                    full_response += delta
                                    yield sse({"type": "text", "content": delta})
                tools_used = extract_tool_calls(run.result)
                if tools_used:
                    yield sse({"type": "tools", "tools": [t.model_dump() for t in tools_used]})
                await add_message(
                    session_id,
                    "assistant",
                    full_response,
                    {"streamed": True, "tool_calls": len(tools_used)},
                )
                yield sse({"type": "end"})
            except Exception:
                logger.exception("Stream failed")
                yield sse({"type": "error", "content": "The agent could not process the request"})

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/search/vector", response_model=SearchResponse, dependencies=protected)
    async def search_vector(request: SearchRequest) -> SearchResponse:
        started = datetime.now()
        try:
            results = await vector_search_tool(
                VectorSearchInput(query=request.query, limit=request.limit)
            )
        except Exception as exc:
            logger.exception("Vector search failed")
            raise HTTPException(status_code=502, detail="Search failed") from exc
        return SearchResponse(
            results=results,
            total_results=len(results),
            search_type=SearchType.VECTOR,
            query_time_ms=(datetime.now() - started).total_seconds() * 1000,
        )

    @app.post("/search/hybrid", response_model=SearchResponse, dependencies=protected)
    async def search_hybrid(request: SearchRequest) -> SearchResponse:
        started = datetime.now()
        try:
            results = await hybrid_search_tool(
                HybridSearchInput(query=request.query, limit=request.limit, text_weight=0.3)
            )
        except Exception as exc:
            logger.exception("Hybrid search failed")
            raise HTTPException(status_code=502, detail="Search failed") from exc
        return SearchResponse(
            results=results,
            total_results=len(results),
            search_type=SearchType.HYBRID,
            query_time_ms=(datetime.now() - started).total_seconds() * 1000,
        )

    @app.get("/documents", dependencies=protected)
    async def list_documents_endpoint(
        limit: int = Query(default=20, ge=1, le=100), offset: int = Query(default=0, ge=0)
    ) -> dict[str, Any]:
        try:
            documents = await list_documents_tool(DocumentListInput(limit=limit, offset=offset))
        except Exception as exc:
            logger.exception("Document listing failed")
            raise HTTPException(status_code=502, detail="Document listing failed") from exc
        return {"documents": documents, "total": len(documents), "limit": limit, "offset": offset}

    @app.get("/sessions/{session_id}", dependencies=protected)
    async def get_session_info(session_id: str, user_id: str | None = None) -> dict[str, Any]:
        try:
            uuid.UUID(session_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail="Session not found") from exc
        session = await get_session(session_id)
        if not session or session.get("user_id") != user_id:
            raise HTTPException(status_code=404, detail="Session not found")
        return session

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        request_id = str(uuid.uuid4())
        logger.error("Unhandled exception [%s]: %s", request_id, exc, exc_info=exc)
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "error_type": "InternalError",
                "request_id": request_id,
            },
        )


app = create_app()


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "agent.api:app",
        host=settings.APP_HOST,
        port=settings.APP_PORT,
        reload=not settings.is_production,
        log_level=settings.LOG_LEVEL.lower(),
    )

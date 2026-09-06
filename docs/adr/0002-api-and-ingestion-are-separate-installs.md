# ADR 0002 — The API and the ingestion pipeline are separate installs

**Status:** accepted

## Context

`pyproject.toml` listed Docling, LangChain, Streamlit and `pydantic-ai` (the
full package, every provider) as dependencies of one project, and the single
`Dockerfile` installed all of it into every image — the API container carried
a PDF layout model and a CUDA torch it never imports. The Dockerfile used
`python:3.11` for a project that declares `>=3.12,<3.13`. `pydantic-ai`'s
`result.data` had become `result.output` in the pinned version.

Modules also did work at import: the OpenAI embedding client was built when
`agent/tools.py` was imported (so `agent.api` needed a key to import at all),
the agent was constructed at import, and Docling was imported at the top of
the extractor module. `providers.get_llm_model` defaulted the API key to the
literal string `"LLM_API_KEY"`.

## Decision

- The project's `dependencies` are the API's: FastAPI, asyncpg,
  `pydantic-ai-slim[openai]`, `pydantic-settings`. Docling and LangChain are
  the `ingest` extra; Streamlit and its HTTP clients the `ui` extra (they were
  undeclared before and worked by accident). `torch` resolves from the CPU
  index via `[tool.uv.index]`.
- One `Dockerfile`, `python:3.12-slim`, non-root, with a build arg that
  selects the extra. Compose builds the API, the UI and — under a profile —
  an ingestion runner.
- Model, embedding client and agent are created on first use through
  `get_*` functions with a `set_agent`/`reset_clients` seam for tests; the
  extractor imports Docling when constructed and raises a clear error if it
  is missing instead of leaving a half-built object.
- `Settings` (pydantic-settings) replaces the scattered `os.getenv` calls,
  including the mis-typed default.

## Consequences

- The API image is small and starts without secrets present; the tests run
  without PostgreSQL, OpenAI or Docling.
- Ingestion is an explicit, heavier install (`uv sync --extra ingest`),
  exercised by its own CI job.
- Operators pick the interpreter the project actually supports.

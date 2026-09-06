# Threat model

Scope: a FastAPI service that runs a Pydantic AI agent with pgvector search
tools over ingested PDFs, a Streamlit chat client, and an offline ingestion
pipeline (Docling + OpenAI embeddings) that writes to PostgreSQL. Deployed
with Docker Compose.

## What it holds

| Asset | Where | Why it matters |
|---|---|---|
| `OPENAI_API_KEY` | environment | Every chat, search and ingestion spends it |
| `API_KEY` | environment | Gates the whole API |
| The document corpus and its embeddings | PostgreSQL | Whatever was ingested — here, sustainability reports |
| Conversations | `sessions`/`messages` tables | Whatever users asked |
| `DB_PASSWORD` | environment / compose | The database |

## Threats

### T1 — Anonymous use *(was open)*

Every route was unauthenticated and CORS was `*` with credentials.
**Controls.** `X-API-Key` on every route but `/health`; 503 when no key is
configured; explicit `CORS_ORIGINS`. See [ADR 0001](adr/0001-fail-closed-auth-and-caller-scoped-sessions.md).
**Residual.** One shared key: no per-user identity, no per-user quota.

### T2 — Session cross-talk *(was open)*

Any caller could read or continue any session by id. **Controls.** Sessions
are bound to the declared `user_id`; mismatches are 404. **Residual.** The
identity is asserted by the caller who holds the key.

### T3 — Information leaks in errors *(was open)*

`detail=str(e)` on every route and a global handler that returned
`str(exc)` (and was not a `Response`, so it crashed as well). **Controls.**
Generic messages with a request id; details in the server log only; the
agent's failure is a 502, not a saved assistant turn containing the
exception.

### T4 — Prompt injection through the corpus

Retrieved chunks and whole documents are fed to the model. **Controls.** The
agent's tools are read-only searches; there is no tool with side effects.
**Residual.** A poisoned document can steer answers; ingestion is an
operator action, not a user upload.

### T5 — Resource exhaustion

`limit`/`offset` on `/documents` and the search inputs were unbounded and
went to SQL; messages had no length limit. **Controls.** Bounded Pydantic
models and query parameters (`limit ≤ 50` for search, `≤ 100` for listing,
message `≤ 8000` chars); `MAX_CONTEXT_MESSAGES`. **Residual.** No rate limit
per key.

### T6 — SQL

Queries use asyncpg parameters and SQL functions (`match_chunks`,
`hybrid_search`). One `LIMIT` was interpolated with an f-string from an
internal integer; it is a parameter now. Document ids are validated as UUIDs
before `::uuid` casts.

### T7 — Container and database exposure

The image ran as root on Python 3.11; compose published PostgreSQL on the
host with the default password. **Controls.** Non-root uid 10001, Python
3.12, `DB_PASSWORD` required by compose, PostgreSQL not published by
default. **Residual.** TLS termination and network policy belong to the
deployment.

### T8 — Supply chain

Docling/torch were installed into the API image. **Controls.** Dependency
groups, CPU-only torch index, `pip-audit`, `bandit` and gitleaks in CI, a
lockfile.

## Not addressed

- Per-user authentication and quotas (see T1).
- Encryption at rest for the corpus and conversations.
- The Streamlit UI has no login of its own; it is a client that holds the
  API key.

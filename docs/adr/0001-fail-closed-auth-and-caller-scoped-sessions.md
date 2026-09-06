# ADR 0001 — The API fails closed, and sessions belong to the caller who made them

**Status:** accepted

## Context

None of the routes authenticated. `/chat` and `/chat/stream` spend OpenAI
credit per request; `/search/*` spend embedding credit; `/documents` lists
the corpus; `/sessions/{id}` returned any session's metadata by id, and
`/chat` continued any session whose id the caller knew. CORS was
`allow_origins=["*"]` with `allow_credentials=True`. The README's deployment
story was `docker-compose up` with the API port published.

## Decision

- Every route except `/health` requires `X-API-Key`, compared in constant
  time with `API_KEY`. When `API_KEY` is not configured the routes answer
  **503**, not 200: an operator who forgets the key gets a closed API and a
  log line, not an open one.
- `CORS_ORIGINS` is an explicit list; no origins means no cross-origin
  browser access, and credentials are never allowed.
- A session is bound to the `user_id` supplied when it was created. `/chat`
  with someone else's `session_id` — or with no `user_id` for a session that
  has one — is a 404, and so is `/sessions/{id}` unless `?user_id=` matches.
  Identities are caller-asserted: this is a single-tenant API behind one key,
  and the check prevents accidental cross-talk, not a determined caller who
  holds the key.
- `/health` stays open so orchestrators can probe it; it reports whether the
  database answers and whether the model and the API key are configured,
  instead of a hard-coded `llm_connection: true`.

## Consequences

- Nothing spends money or reads data without the key.
- The Streamlit UI sends the key from its own environment; it no longer lets
  its users type an arbitrary API URL for the server to call.
- Per-user tenancy would need real authentication (JWT/OIDC) in front of
  this; the session check is the seam for it.

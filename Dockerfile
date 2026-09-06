# One image, three roles, chosen at build time:
#   API      (default)          docker build .
#   UI       --build-arg EXTRAS="--extra ui"
#   ingest   --build-arg EXTRAS="--extra ingest"   (Docling + CPU torch; large)
# Python 3.12: pyproject declares >=3.12,<3.13; the old image used 3.11.
FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1
WORKDIR /app
COPY --from=ghcr.io/astral-sh/uv:0.8 /uv /usr/local/bin/uv

ARG EXTRAS=""
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project ${EXTRAS}

COPY agent ./agent
COPY ingestion ./ingestion
COPY ui ./ui
COPY sql ./sql

RUN useradd --uid 10001 --create-home --shell /usr/sbin/nologin app \
    && mkdir -p /app/documents && chown -R app:app /app
USER 10001
ENV PATH="/app/.venv/bin:$PATH" \
    APP_HOST=0.0.0.0 \
    APP_PORT=8058
EXPOSE 8058
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8058/health', timeout=4).status == 200 else 1)"
CMD ["sh", "-c", "uvicorn agent.api:app --host ${APP_HOST} --port ${APP_PORT}"]

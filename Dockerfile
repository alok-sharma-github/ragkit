# RAGkit — one image, one service, one URL.
#
# The frontend is built here rather than committed as a bundle, so the deployed
# UI cannot drift from the source it was built from. The Python stage then serves
# that bundle itself (see RAGKIT_WEB_DIST), which is why there is no nginx, no
# second service and no cross-origin configuration to get wrong.

# --- stage 1: build the SPA -------------------------------------------------
FROM node:22-slim AS web

WORKDIR /web
# package files first: this layer is cached unless the dependency set changes,
# so an edit to a .tsx file does not reinstall node_modules.
COPY app/web/package*.json ./
RUN npm ci

COPY app/web/ ./
# `npm run build` is `tsc -b && vite build`. Deliberately NOT `tsc --noEmit`,
# which exits 0 on code this build rejects -- so a type error must fail the
# image, not surface after deploy.
RUN npm run build


# --- stage 2: the app -------------------------------------------------------
FROM python:3.13-slim AS app

# PyMuPDF ships manylinux wheels, so no compiler is needed. curl is here for the
# healthcheck only.
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # The cp1252 console crashed six separate runs on this project (box-drawing
    # characters, em dashes, accented author names, a Turkish dotless i). The
    # package forces UTF-8 at import, and this belts it at the process level for
    # anything that runs before that import.
    PYTHONIOENCODING=utf-8 \
    # Read-only deployment. Set here rather than in the platform dashboard so
    # the image is safe by DEFAULT -- a deploy that forgets to set it is still
    # protected, which is the direction this has to fail in.
    RAGKIT_DEMO_MODE=1 \
    RAGKIT_WEB_DIST=/app/app/web/dist

COPY pyproject.toml uv.lock* ./
# --extra web is REQUIRED, not optional polish: fastapi, uvicorn and
# python-multipart live in the `web` optional-dependency group, so a plain
# `uv sync` produces an image with no web server and the failure appears only at
# container start. The `rerank` extra is deliberately excluded -- it pulls torch
# (~2GB) for a reranker this build does not use (see FUTURE_SCOPE.md B3).
RUN pip install --no-cache-dir uv \
 && uv sync --frozen --no-dev --extra web

# Source, then data. Data changes rarely and is large, so it goes in its own
# layer; code changes constantly and must not force the data to be re-sent.
COPY ragkit/ ./ragkit/
COPY app/api.py ./app/api.py
COPY scripts/ ./scripts/

# The prebuilt index and the corpus travel WITH the image. Without them the
# container boots to an empty demo and needs a 20-minute ingest against a
# free-tier key before it can answer anything -- so "clone and run" would be
# false for the one person the deployment exists for.
COPY data/ ./data/

COPY --from=web /web/dist ./app/web/dist

# Railway injects PORT. Default 8000 so the image also runs locally unchanged.
ENV PORT=8000
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD curl -fsS "http://127.0.0.1:${PORT}/api/health" || exit 1

# One worker on purpose. The index is held in memory per process (~6.5MB of
# vectors plus the parsed corpus), and the rate limiter's counters are per
# process too -- so a second worker would double the memory and halve the
# effective limit without either being visible. Concurrency here is bounded by a
# free-tier Gemini key, not by CPU.
CMD ["sh", "-c", "uv run uvicorn app.api:app --host 0.0.0.0 --port ${PORT} --workers 1"]

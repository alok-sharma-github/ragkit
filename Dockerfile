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
    # Same safe-by-default argument, applied to the WORDING. config defaults this
    # to `internal`, which is right for a developer on a laptop and wrong in an
    # image: a deployment that forgets to set it would hand a public visitor
    # "raise RAGKIT_MAX_OPERATION_TOKENS" -- operator advice to someone who
    # cannot act on it, which is the defect just fixed reappearing as a config
    # default. Customer and internal deployments override this explicitly.
    #
    # Failing toward the over-cautious reading is cheap; failing toward leaking
    # env var names to strangers is not.
    RAGKIT_DEPLOYMENT_KIND=demo \
    # A public demo spends OUR key. The per-operation cap bounds any single
    # answer; this bounds the day. ~500k tokens is roughly 50 answers, which is
    # generous for a demo and about a dollar at flash rates -- versus the 5M
    # default, which could run to three figures a month if fully consumed.
    RAGKIT_DAILY_TOKEN_CAP=500000 \
    RAGKIT_WEB_DIST=/app/app/web/dist \
    # The venv is on PATH so `uvicorn` resolves without `uv run`, and /app is on
    # PYTHONPATH so `ragkit` and `app` import as top-level packages. Set
    # explicitly rather than relying on the working directory landing on
    # sys.path -- that behaviour differs between how a server is launched, and a
    # silent ImportError at container start is a bad way to learn which.
    PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH=/app

COPY pyproject.toml uv.lock* ./
# --extra web is REQUIRED, not optional polish: fastapi, uvicorn and
# python-multipart live in the `web` optional-dependency group, so a plain
# `uv sync` produces an image with no web server and the failure appears only at
# container start. The `rerank` extra is deliberately excluded -- it pulls torch
# (~2GB) for a reranker this build does not use (see FUTURE_SCOPE.md B3).
#
# --no-install-project installs the DEPENDENCIES only, not this repo as a
# package. Two reasons:
#
#   it failed without it   pyproject declares `readme = "README.md"`, so
#                          hatchling tried to build the local package in a layer
#                          where only pyproject.toml and uv.lock had been copied.
#                          "Readme file does not exist" -- a build error caused
#                          entirely by layer ordering, not by anything wrong.
#   the layer stays cached the dependency layer no longer depends on README.md or
#                          any source file, so editing prose or code does not
#                          reinstall numpy and PyMuPDF.
#
# The code is copied in below and imported from /app directly (PYTHONPATH), which
# is what `uv run` was doing anyway.
RUN pip install --no-cache-dir uv \
 && uv sync --frozen --no-dev --extra web --no-install-project

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

# The app writes conversations to data/index/conversations/ and feedback to
# data/eval/feedback.jsonl at runtime. Hugging Face Spaces does not guarantee the
# container runs as root, and a non-root UID against a root-owned directory turns
# every conversation into a 500 -- so widen it rather than assume the UID.
#
# These writes are EPHEMERAL on any scale-to-zero host: conversations and
# feedback reset when the container restarts. Acceptable for a demo, and stated
# here so nobody later reads a lost conversation as data loss.
RUN chmod -R 0777 /app/data

# Railway injects PORT. Default 8000 so the image also runs locally unchanged.
ENV PORT=8000
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD curl -fsS "http://127.0.0.1:${PORT}/api/health" || exit 1

# One worker on purpose. The index is held in memory per process (measured 193MB
# RSS through the serving path), and the rate limiter's counters are per process
# too -- so a second worker would double the memory and halve the effective limit
# without either being visible. Concurrency here is bounded by a free-tier Gemini
# key, not by CPU.
#
# uvicorn is invoked DIRECTLY from the venv rather than through `uv run`. A
# package manager at container start is a resolver step, a network dependency and
# a class of startup failure that has nothing to do with the application -- and
# on a 0.25 vCPU instance it is also dead time on every restart.
CMD ["sh", "-c", "uvicorn app.api:app --host 0.0.0.0 --port ${PORT} --workers 1"]

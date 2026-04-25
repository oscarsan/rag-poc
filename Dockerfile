# syntax=docker/dockerfile:1.7

FROM python:3.11-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /build

RUN pip install --no-cache-dir uv

# Copy only pyproject.toml here. README.md and app/ are deliberately
# copied later, so README edits don't bust this expensive
# dependency-install layer (torch + sentence-transformers ~3 GB).
# uv pip compile only reads [project.dependencies] from pyproject.toml,
# it does not invoke the build backend, so README.md is not needed yet.
COPY pyproject.toml ./

RUN uv venv /opt/venv \
 && uv pip compile pyproject.toml -o requirements.txt \
 && VIRTUAL_ENV=/opt/venv uv pip install --no-cache -r requirements.txt

# README.md is needed by hatchling when building the project itself
# (referenced from [project].readme). Copy it alongside the source so the
# small final install layer is the only thing that rebuilds on doc/code
# edits.
COPY README.md ./
COPY app ./app

RUN VIRTUAL_ENV=/opt/venv uv pip install --no-cache --no-deps .


FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    HF_HOME=/app/.cache/huggingface

RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/* \
 && groupadd --system app \
 && useradd --system --gid app --home /app --shell /sbin/nologin app

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY app ./app
COPY content ./content

RUN mkdir -p /app/.cache && chown -R app:app /app

USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -fsS http://127.0.0.1:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

# Running the Resort RAG Chatbot locally on macOS

Operational runbook for bringing this PoC up on a Mac. Tested on macOS 14
(Sonoma) and 15 (Sequoia), on both Apple Silicon (arm64) and Intel
(x86_64). Two run modes are covered:

- **A. All-in-Docker** (recommended for ops / demos) — Qdrant + API both in
  containers, orchestrated by `docker compose`.
- **B. Hybrid** (convenient for development) — Qdrant in Docker, API run
  directly from a local virtualenv (also lets the embedding model use
  Apple Silicon's MPS acceleration, which container builds can't).

If you just want to see it working, follow mode A end-to-end. Use mode B
when you're iterating on the code or want MPS acceleration on Apple
Silicon.

---

## 1. System prerequisites

### 1.1 Xcode Command Line Tools

Needed for git and for native Python wheels to build. Skip if you already
have them.

```bash
xcode-select --install
```

### 1.2 Homebrew

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

After install, follow the hint Homebrew prints to add it to your PATH
(typical on Apple Silicon: add `eval "$(/opt/homebrew/bin/brew shellenv)"`
to `~/.zprofile`).

Helpful CLI extras used below:

```bash
brew install git curl jq make
```

### 1.3 Docker Desktop for Mac

Download from <https://www.docker.com/products/docker-desktop/> (or
`brew install --cask docker`) and launch the app at least once so it
starts its VM. Verify:

```bash
docker version
docker compose version
```

**Allocate enough resources.** Docker Desktop → Settings → Resources:

- CPUs: 4 or more
- Memory: **at least 6 GB** (bge-m3 is ~2 GB and torch needs headroom)
- Disk: 30 GB or more (model + images + volumes)

Defaults on fresh installs are often too small and manifest as OOM kills
during ingestion.

### 1.4 Python 3.11 (only for mode B or to run tests/ingest locally)

Easiest path is Homebrew:

```bash
brew install python@3.11
# Homebrew does not put it on PATH as `python3` by default; use the full
# path or symlink. The Makefile uses `uv venv --python 3.11`, which finds
# it automatically.
/opt/homebrew/opt/python@3.11/bin/python3.11 --version   # Apple Silicon
/usr/local/opt/python@3.11/bin/python3.11 --version      # Intel
```

Alternative: `pyenv install 3.11` if you prefer pyenv.

### 1.5 uv (Python dep manager)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# reload shell so uv is on PATH:
exec $SHELL -l
uv --version
```

Or: `brew install uv`.

---

## 2. Get the code

```bash
git clone <your-repo-url> rag-poc
cd rag-poc
git checkout claude/resort-rag-chatbot-JTZce
```

---

## 3. Configure

```bash
cp .env.example .env
```

Edit `.env` and at minimum set:

```
ANTHROPIC_API_KEY=sk-ant-...your-real-key...
```

All other values have sensible defaults for a local run. Notable ones:

| Var                | Default                     | When to change                                              |
|--------------------|-----------------------------|-------------------------------------------------------------|
| `CLAUDE_MODEL`     | `claude-haiku-4-5`          | Swap to another Claude model ID                             |
| `EMBEDDING_MODEL`  | `BAAI/bge-m3`               | Only if you're swapping the embedding model + dimension     |
| `EMBEDDING_DIM`    | `1024`                      | Must match the embedding model above                        |
| `QDRANT_URL`       | `http://localhost:6333`     | Mode A from host: leave as-is. Inside `app` container: compose overrides this to `http://qdrant:6333` automatically |
| `QDRANT_COLLECTION`| `syote`                     | Isolate test data                                           |
| `TOP_K`            | `5`                         | Retrieval breadth                                           |
| `LOG_LEVEL`        | `INFO`                      | `DEBUG` while troubleshooting                               |

**Never commit `.env`** — `.gitignore` already excludes it.

On your editor of choice: `open -e .env` opens it in TextEdit, or use `vim`,
`code .env`, etc.

---

## 4. Mode A — run everything in Docker

### 4.1 Start the stack

```bash
make up
# equivalent to: docker compose up -d --build
```

First build pulls Python 3.11-slim, Qdrant, and installs Python deps
(`torch`, `sentence-transformers`, etc.). On Apple Silicon the arm64
variants are pulled automatically. Expect ~5–10 minutes on first run
(torch + friends are large wheels); subsequent starts are ~5 seconds.

Check containers:

```bash
docker compose ps
```

You should see both `rag-poc-qdrant` and `rag-poc-app` as `running`
(healthy). Tail logs:

```bash
make logs
# or: docker compose logs -f app qdrant
```

### 4.2 Ingest content

The app is up, but Qdrant is empty. Run the ingestion CLI inside the `app`
container so it reuses the same virtualenv and reaches Qdrant via the
compose network:

```bash
docker compose exec app python -m app.ingestion.cli ingest --path content/syote
```

**First run is slow.** The ingest downloads `BAAI/bge-m3` from Hugging Face
(~2.2 GB) into the `hf_cache` Docker volume. Subsequent runs reuse the
cache and complete in seconds.

Expected output ends with a line like:

```
Done. files=7 chunks=14 upserted=14 errors=0
```

Re-running is idempotent — chunks with the same `{doc_id}__{lang}__{idx}`
are overwritten, not duplicated.

### 4.3 Smoke-test the API

```bash
curl -s http://localhost:8000/health | jq
# -> {"status":"ok"}

curl -s -X POST http://localhost:8000/chat \
  -H 'content-type: application/json' \
  -d '{"message":"How much is the husky safari?"}' | jq

curl -s -X POST http://localhost:8000/chat \
  -H 'content-type: application/json' \
  -d '{"message":"Paljonko husky safari maksaa?"}' | jq
```

You should see a grounded answer quoting `159 €` and `sources:
["husky-safari-2h"]`.

### 4.4 Stop

```bash
make down
# equivalent to: docker compose down
```

This stops the containers but keeps the `qdrant_storage` and `hf_cache`
volumes, so the next start is fast and ingested data persists. To wipe
data as well:

```bash
docker compose down -v
```

---

## 5. Mode B — hybrid (Qdrant in Docker, API local)

Useful when you want auto-reload on code changes, want to debug the app
from VS Code / PyCharm, or want MPS acceleration on Apple Silicon (which
the containerised build can't use).

### 5.1 Start only Qdrant

```bash
docker compose up -d qdrant
docker compose ps qdrant
```

### 5.2 Install the app locally

```bash
make install
# creates .venv with Python 3.11 and installs the project + dev deps
```

On Apple Silicon, `torch` installs as the native arm64 build, which will
use MPS automatically when available. `sentence-transformers` picks this
up via `SentenceTransformer(..., device=None)`.

### 5.3 Ingest

```bash
make ingest
# equivalent to: .venv/bin/python -m app.ingestion.cli ingest --path content/syote
```

First run downloads `bge-m3` into `~/.cache/huggingface` (~2.2 GB). On
Apple Silicon MPS, ingestion of the sample content takes a few seconds
after the model is cached.

### 5.4 Run the API with reload

```bash
make run
# equivalent to: .venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Smoke-test the same way as in §4.3. Ctrl-C to stop.

### 5.5 Tests and lint

```bash
make test    # pytest
make lint    # ruff
```

---

## 6. Day-to-day operations

### 6.1 Updating content

1. Edit or add files under `content/syote/`.
2. Re-run ingestion (§4.2 or §5.3). Idempotent.
3. No need to restart the app — it reads from Qdrant live.

### 6.2 Rotating the Claude API key

1. Update `ANTHROPIC_API_KEY` in `.env`.
2. Mode A: `docker compose up -d` (compose will recreate `app` with the
   new env). Mode B: restart `make run`.

### 6.3 Changing the embedding model

The Qdrant collection is tied to the embedding dimension. If you change
`EMBEDDING_MODEL` to something with a different `EMBEDDING_DIM`:

1. Update both vars in `.env`.
2. Re-ingest. `QdrantStore.ensure_collection` detects the dimension
   mismatch, drops the collection, and recreates it.
3. All prior points are lost. This is intentional — old vectors are
   incompatible with the new model.

### 6.4 Viewing what's in Qdrant

Qdrant exposes a web dashboard at <http://localhost:6333/dashboard>.
Useful for browsing collections, inspecting payloads, and sanity-checking
ingestion.

### 6.5 Log levels

Set `LOG_LEVEL=DEBUG` in `.env` and restart. `RagService` will log the
retrieval language, `top_k`, and the number of chunks returned per query.

---

## 7. Troubleshooting

### `Cannot connect to the Docker daemon`

Docker Desktop isn't running. Open `/Applications/Docker.app` and wait
for the whale icon in the menu bar to stop animating. `docker version`
should then succeed.

### `bind: address already in use` on 6333 or 8000

Something else is listening. Find it:

```bash
sudo lsof -iTCP:6333 -sTCP:LISTEN
sudo lsof -iTCP:8000 -sTCP:LISTEN
```

Stop the conflicting process, or change the host-side port in
`docker-compose.yml` (e.g. `"6334:6333"`).

### App container restarts repeatedly

```bash
docker compose logs app --tail=200
```

Most common causes:

- **`ANTHROPIC_API_KEY` is empty** — the `ClaudeProvider` constructor
  raises. Set the key in `.env` and recreate the container.
- **Qdrant not reachable** — the compose healthcheck should gate this, but
  if you overrode `QDRANT_URL` manually, make sure it points at
  `http://qdrant:6333` from inside the container.

### Ingestion is killed / OOM on first run

Docker Desktop memory is too low. Open Docker Desktop → Settings →
Resources → increase Memory to at least 6 GB, apply & restart. Then
re-run ingestion.

### Ingestion hangs on "Loading embedding model"

It's downloading `bge-m3` (~2.2 GB) from Hugging Face. First run only.
Check network from the container:

```bash
docker compose exec app curl -s -o /dev/null -w '%{http_code}\n' https://huggingface.co
```

Expected: `200`. If you're on a VPN or corporate proxy, export
`HTTPS_PROXY` into the container via `docker-compose.yml`.

### `Dimension mismatch` warning in logs, and searches return nothing

You changed `EMBEDDING_MODEL` but didn't re-ingest. Run ingestion again —
`ensure_collection` will recreate the collection at the new dimension.

### `make install` fails building wheels

Ensure Xcode Command Line Tools are present (`xcode-select --install`)
and that `uv` is finding Python 3.11 (`uv python list`). On Apple Silicon
make sure Homebrew is installed to `/opt/homebrew` and on your PATH — if
you accidentally have an Intel-under-Rosetta shell, `uv` may pick the
wrong Python.

### Tests fail with `ModuleNotFoundError: app`

You ran `pytest` without the venv. Use `make test` or
`.venv/bin/pytest -v`.

### Volume mount "not shared" error

Docker Desktop → Settings → Resources → File Sharing. Ensure the path
you cloned the repo to (typically under `$HOME`) is in the allowed list.
`$HOME` is shared by default on recent Docker Desktop versions.

---

## 8. Tear-down

```bash
# stop containers, keep data and caches
docker compose down

# stop and delete Qdrant data + HF model cache volumes
docker compose down -v

# remove local venv too
rm -rf .venv

# optional: reclaim disk from Docker images/layers
docker system prune -a
```

That fully cleans the machine of PoC state (the `.env` file and code
remain on disk).

---

## 9. Security reminder

This PoC is **not** hardened for exposure beyond localhost:

- CORS is `allow_origins=["*"]`.
- No auth on `/chat`.
- No rate limiting or request-size caps.
- The example on-call phone number in
  `content/syote/practical/check-in-out.md` is a placeholder.

Treat any bind beyond `127.0.0.1` as a staging/production decision that
requires the items above to be addressed first.

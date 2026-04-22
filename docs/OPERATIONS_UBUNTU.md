# Running the Resort RAG Chatbot locally on Ubuntu

Operational runbook for bringing this PoC up on an Ubuntu machine (tested
against 22.04 LTS and 24.04 LTS). Two run modes are covered:

- **A. All-in-Docker** (recommended for ops / demos) — Qdrant + API both in
  containers, orchestrated by `docker compose`.
- **B. Hybrid** (convenient for development) — Qdrant in Docker, API run
  directly from a local virtualenv.

If you just want to see it working, follow mode A end-to-end. Use mode B
when you're iterating on the code.

---

## 1. System prerequisites

Run once, as a sudoer on the Ubuntu host.

### 1.1 Base packages

```bash
sudo apt update
sudo apt install -y git curl jq make build-essential ca-certificates
```

### 1.2 Docker Engine + Compose plugin

The `docker.io` package from the default Ubuntu repos works but is often old
— prefer Docker's own apt repo:

```bash
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
  | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo \"$VERSION_CODENAME\") stable" \
  | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

Let your user run Docker without sudo (log out and back in after):

```bash
sudo usermod -aG docker "$USER"
```

Verify:

```bash
docker version
docker compose version
```

### 1.3 Python 3.11 (only needed for mode B or to run tests/ingest locally)

Ubuntu 24.04 ships `python3.12`; this project targets 3.11. Use the
deadsnakes PPA:

```bash
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3.11-dev
python3.11 --version
```

On 22.04, `python3.11` is available from the same PPA.

### 1.4 uv (Python dep manager)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# reload shell so uv is on PATH:
exec $SHELL -l
uv --version
```

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

---

## 4. Mode A — run everything in Docker

### 4.1 Start the stack

```bash
make up
# equivalent to: docker compose up -d --build
```

First build pulls Python 3.11-slim, Qdrant, and installs Python deps
(`torch`, `sentence-transformers`, etc.). Expect ~3–6 minutes on a decent
connection. Subsequent starts are ~5 seconds.

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

Useful when you want auto-reload on code changes, or when debugging the app
in an IDE.

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

### 5.3 Ingest

```bash
make ingest
# equivalent to: .venv/bin/python -m app.ingestion.cli ingest --path content/syote
```

First run downloads `bge-m3` into `~/.cache/huggingface` (~2.2 GB).

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
2. Mode A: `docker compose up -d` (compose will recreate `app` with the new
   env). Mode B: restart `make run`.

### 6.3 Changing the embedding model

The Qdrant collection is tied to the embedding dimension. If you change
`EMBEDDING_MODEL` to something with a different `EMBEDDING_DIM`:

1. Update both vars in `.env`.
2. Re-ingest. `QdrantStore.ensure_collection` detects the dimension
   mismatch, drops the collection, and recreates it.
3. All prior points are lost. This is intentional — old vectors are
   incompatible with the new model.

### 6.4 Viewing what's in Qdrant

Qdrant exposes a web dashboard at <http://localhost:6333/dashboard>. Useful
for browsing collections, inspecting payloads, and sanity-checking
ingestion.

### 6.5 Log levels

Set `LOG_LEVEL=DEBUG` in `.env` and restart. `RagService` will log the
retrieval language, `top_k`, and the number of chunks returned per query.

---

## 7. Troubleshooting

### `permission denied while trying to connect to the Docker daemon socket`

You haven't logged out/in since `usermod -aG docker`. Run `newgrp docker`
in the current shell, or log out and back in.

### `bind: address already in use` on 6333 or 8000

Something else is listening. Find it:

```bash
sudo ss -ltnp | grep -E ':(6333|8000)\b'
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

### Ingestion hangs on "Loading embedding model"

It's downloading `bge-m3` (~2.2 GB) from Hugging Face. First run only.
Check network:

```bash
docker compose exec app curl -s -o /dev/null -w '%{http_code}\n' https://huggingface.co
```

Expected: `200`. If you're behind a corporate proxy, export `HTTPS_PROXY`
into the container via `docker-compose.yml`.

### `Dimension mismatch` warning in logs, and searches return nothing

You changed `EMBEDDING_MODEL` but didn't re-ingest. Run ingestion again —
`ensure_collection` will recreate the collection at the new dimension.

### OOM during embedding

`bge-m3` is ~2 GB in memory. On very small VMs, reduce the ingestion batch
size (pass it via the `IngestionPipeline` constructor) or use a smaller
embedding model and update `EMBEDDING_DIM` to match.

### Tests fail with `ModuleNotFoundError: app`

You ran `pytest` without the venv. Use `make test` or
`.venv/bin/pytest -v`.

---

## 8. Tear-down

```bash
# stop containers, keep data and caches
docker compose down

# stop and delete Qdrant data + HF model cache volumes
docker compose down -v

# remove local venv too
rm -rf .venv
```

That fully cleans the machine of PoC state (the `.env` file and code remain
on disk).

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

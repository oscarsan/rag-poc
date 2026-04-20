.PHONY: help install up down logs ingest test lint run

help:
	@echo "Targets:"
	@echo "  install   Create venv and install project + dev deps via uv"
	@echo "  up        Start qdrant + app via docker-compose"
	@echo "  down      Stop and remove docker-compose services"
	@echo "  logs      Tail docker-compose logs"
	@echo "  run       Run the API locally (requires qdrant reachable at QDRANT_URL)"
	@echo "  ingest    Run the ingestion CLI against content/syote"
	@echo "  test      Run pytest"
	@echo "  lint      Run ruff"

install:
	uv venv --python 3.11
	uv pip install -e ".[dev]"

up:
	docker compose up -d --build

down:
	docker compose down

logs:
	docker compose logs -f --tail=200

run:
	.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

ingest:
	.venv/bin/python -m app.ingestion.cli ingest --path content/syote

test:
	.venv/bin/pytest -v

lint:
	.venv/bin/ruff check app tests

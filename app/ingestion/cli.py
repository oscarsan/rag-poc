from __future__ import annotations

import logging
from pathlib import Path

import typer

from app.config import configure_logging, get_settings
from app.ingestion.pipeline import IngestionPipeline
from app.providers.embeddings import BgeM3EmbeddingProvider
from app.providers.vectorstore import QdrantStore

app = typer.Typer(help="Ingestion CLI for the resort RAG PoC.", add_completion=False)
log = logging.getLogger(__name__)


@app.command()
def ingest(
    path: Path = typer.Option(
        Path("content/syote"),
        "--path",
        "-p",
        help="Root directory of markdown content to ingest.",
    ),
    max_words: int = typer.Option(400, help="Max words per chunk."),
) -> None:
    """Walk PATH, parse markdown, chunk, embed, and upsert into Qdrant."""
    settings = get_settings()
    configure_logging(settings.log_level)

    log.info("Ingesting from %s", path)
    embeddings = BgeM3EmbeddingProvider(model_name=settings.embedding_model)
    store = QdrantStore(url=settings.qdrant_url, collection=settings.qdrant_collection)
    pipeline = IngestionPipeline(embeddings, store, max_words=max_words)
    report = pipeline.run(path)

    typer.echo(
        f"Done. files={report.files_processed} "
        f"chunks={report.chunks_created} upserted={report.points_upserted} "
        f"errors={len(report.errors)}"
    )
    for err in report.errors:
        typer.echo(f"  ERROR: {err}", err=True)


def main() -> None:
    app()


if __name__ == "__main__":
    main()

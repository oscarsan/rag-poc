from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import router
from app.config import configure_logging, get_settings


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level)

    app = FastAPI(
        title="Syöte Resort RAG Chatbot",
        version="0.1.0",
        description="PoC RAG chatbot for a Finnish resort. PoC CORS: permissive. "
        "Lock this down before any non-local deployment.",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
        allow_credentials=False,
    )
    app.include_router(router)
    return app


app = create_app()

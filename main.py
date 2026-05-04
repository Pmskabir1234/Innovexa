"""FastAPI entrypoint for the technical decision assistant."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router
from api.stream_routes import router as stream_router
from services.assistant import TechnicalAssistant
from utils.config import ensure_runtime_dirs, get_settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Train models on startup so first request is warm."""

    settings = ensure_runtime_dirs(get_settings())
    assistant = TechnicalAssistant(settings)
    assistant.fit_from_synthetic()
    app.state.assistant = assistant
    yield


app = FastAPI(
    title="Smart AI Assistant for Technical Decision-Making",
    version="0.1.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router)
app.include_router(stream_router)


@app.get("/health")
def health() -> dict:
    """Lightweight readiness probe."""

    return {"status": "ok"}

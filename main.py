"""FastAPI entrypoint for the technical decision assistant."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import logging

from api.routes import router
from api.stream_routes import router as stream_router
from services.assistant import TechnicalAssistant
from utils.config import ensure_runtime_dirs, get_settings

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("assistant")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Train models on startup so first request is warm."""

    settings = ensure_runtime_dirs(get_settings())
    assistant = TechnicalAssistant(settings)
    assistant.fit_from_synthetic()
    app.state.assistant = assistant
    
    # Log endpoint registration
    logger.info("Registering endpoints:")
    for route in app.routes:
        if hasattr(route, "methods"):
            logger.info(f"  {route.methods} {route.path}")
            
    yield


app = FastAPI(
    title="Smart AI Assistant for Technical Decision-Making",
    version="0.1.0",
    lifespan=lifespan,
)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Log detailed validation errors."""
    logger.error(f"Validation error for {request.method} {request.url}: {exc.errors()}")
    logger.error(f"Request body: {await request.body()}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": exc.errors(), "body": str(await request.body())},
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Log any unhandled exception."""
    logger.error(f"Unhandled error for {request.method} {request.url}: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal Server Error", "error": str(exc)},
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


import os
from fastapi.staticfiles import StaticFiles

# Mount React frontend static files if they have been built
frontend_dist = os.path.join(os.path.dirname(__file__), "frontend", "dist")
if os.path.exists(frontend_dist):
    app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")


@app.get("/health")
def health() -> dict:
    """Lightweight readiness probe."""

    return {"status": "ok"}


"""
main.py – Application entry point for the Intelligent Merchant Support &
          Operations platform.

This module creates the FastAPI application instance, mounts all API routers,
and starts the Uvicorn server when executed directly.
"""

import os
from contextlib import asynccontextmanager

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI

# Load environment variables from .env (if present)
load_dotenv()

# Import routers from the api package
from api.chat import router as chat_router
from api.telemetry import router as telemetry_router
from api.webhooks import router as webhooks_router
from data.data_access import DataLoader

# ──────────────────────────────────────────────────────────────────────────────
# Application lifespan – runs once on startup and teardown
# ──────────────────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initialise shared resources on startup and release them on shutdown.

    The ``DataLoader`` reads the three telemetry CSVs into memory once so
    every request can query them without repeated I/O.  If the CSV files are
    missing (e.g. first run before ``data/telemetry_generator.py`` has been
    executed), the DataLoader logs a warning and the app still starts with
    empty DataFrames.
    """
    app.state.data_loader = DataLoader()
    yield
    # Nothing to clean up — in-memory DataFrames are garbage-collected


# ──────────────────────────────────────────────────────────────────────────────
# Application factory
# ──────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Intelligent Merchant Support & Operations",
    description=(
        "Agentic AI platform for diagnosing payment failures, "
        "handling merchant onboarding (KYB/KYC), and dynamically "
        "tuning fraud risk rules."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

# ──────────────────────────────────────────────────────────────────────────────
# Mount routers
# ──────────────────────────────────────────────────────────────────────────────

# Payment event webhooks  →  /webhook/...
app.include_router(webhooks_router, prefix="/webhook", tags=["Webhooks"])

# Merchant support chat   →  /chat/...
app.include_router(chat_router, prefix="/chat", tags=["Chat"])

# Telemetry data API      →  /api/v1/...
app.include_router(telemetry_router, prefix="/api/v1", tags=["Telemetry"])


# ──────────────────────────────────────────────────────────────────────────────
# Root health-check endpoint
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
def root() -> dict:
    """Simple health-check endpoint."""
    return {"status": "ok", "service": "merchant-ai-ops"}


# ──────────────────────────────────────────────────────────────────────────────
# Dev-mode entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=os.getenv("APP_HOST", "0.0.0.0"),
        port=int(os.getenv("APP_PORT", "8000")),
        reload=os.getenv("APP_RELOAD", "true").lower() == "true",
    )


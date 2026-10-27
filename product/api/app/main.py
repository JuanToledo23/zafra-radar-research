"""Zafra Radar API skeleton: /health and /version only. The data endpoints are built during
the hackathon (product/HACKATHON_PLAN.md)."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

from app.db import engine

VERSION = "0.1.0"

app = FastAPI(title="Zafra Radar API", version=VERSION)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET"])


@app.get("/health")
async def health() -> dict:
    """Liveness plus a database round trip, so the healthcheck fails if PostGIS is down."""
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        db = "ok"
    except Exception as exc:  # reported in the body, not raised
        db = f"error: {type(exc).__name__}"
    return {"status": "ok" if db == "ok" else "degraded", "db": db}


@app.get("/version")
async def version() -> dict:
    return {"name": "zafra-radar-api", "version": VERSION}

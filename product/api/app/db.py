"""Async SQLAlchemy engine for the PostGIS database (connection settings from POSTGRES_*)."""

import os

from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine


def database_url() -> str:
    user = os.environ.get("POSTGRES_USER", "zafra")
    password = os.environ.get("POSTGRES_PASSWORD", "zafra_local_only")  # local-only default
    host = os.environ.get("POSTGRES_HOST", "db")
    port = os.environ.get("POSTGRES_PORT", "5432")
    name = os.environ.get("POSTGRES_DB", "zafra")
    return f"postgresql+asyncpg://{user}:{password}@{host}:{port}/{name}"


engine: AsyncEngine = create_async_engine(database_url(), pool_pre_ping=True)

from __future__ import annotations

import sqlite3
from collections.abc import AsyncIterator

import sqlalchemy as sa
from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from backend.config import get_settings
from backend.db.models import Base

settings = get_settings()

# SQLite defaults are hostile to a job engine with 20+ concurrent async
# writers hammering CompanyInput/Job rows: the default rollback-journal mode
# locks the *whole* file for the duration of any write, and busy_timeout
# defaults to 0ms, so a second writer that shows up while another write is
# in flight fails IMMEDIATELY with "database is locked" instead of waiting.
# That crashed real job runs mid-flight (see Job.error_message history).
# WAL mode lets readers proceed concurrently with a single writer, and a
# generous busy_timeout makes writers queue/retry instead of erroring out.
_IS_SQLITE = settings.DATABASE_URL.startswith("sqlite")

engine = create_async_engine(settings.DATABASE_URL, echo=False, future=True)

if _IS_SQLITE:

    @event.listens_for(engine.sync_engine, "connect")
    def _set_sqlite_pragmas(dbapi_connection: sqlite3.Connection, _connection_record: object) -> None:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA busy_timeout=30000")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

SessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


def _add_missing_columns(sync_conn) -> None:
    """`create_all` only creates missing TABLES - a column added to a model
    later never reaches an existing database. Adds any missing (nullable)
    columns in place so existing data (users, API keys) survives model
    changes without a manual migration. Columns removed from a model are
    left untouched in the table; nothing reads them."""
    inspector = sa.inspect(sync_conn)
    existing_tables = set(inspector.get_table_names())
    for table in Base.metadata.sorted_tables:
        if table.name not in existing_tables:
            continue
        have = {c["name"] for c in inspector.get_columns(table.name)}
        for column in table.columns:
            if column.name in have or not column.nullable:
                continue
            col_type = column.type.compile(dialect=sync_conn.dialect)
            sync_conn.execute(sa.text(f'ALTER TABLE "{table.name}" ADD COLUMN "{column.name}" {col_type}'))


async def init_db() -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await conn.run_sync(_add_missing_columns)


async def get_session() -> AsyncIterator[AsyncSession]:
    async with SessionLocal() as session:
        yield session

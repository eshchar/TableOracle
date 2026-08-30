"""SQLite connection handling: sqlite-vec loading, schema migration, vector I/O."""

from __future__ import annotations

import sqlite3
import struct
from collections.abc import Sequence
from pathlib import Path

import sqlite_vec

from tableoracle.config import Settings, get_settings

SCHEMA_PATH = Path(__file__).with_name("schema.sql")


def pack_vector(values: Sequence[float]) -> bytes:
    """Serialize a float vector into the compact form sqlite-vec expects."""
    return struct.pack(f"{len(values)}f", *values)


def unpack_vector(blob: bytes) -> list[float]:
    return list(struct.unpack(f"{len(blob) // 4}f", blob))


def connect(settings: Settings | None = None) -> sqlite3.Connection:
    """Open the index, with the sqlite-vec extension loaded and schema applied.

    The schema is always applied, including on read paths. It is a handful of
    ``CREATE ... IF NOT EXISTS`` statements, and applying it unconditionally
    means a clone that has not been ingested yet answers "no completed ingest"
    from `/healthz` instead of raising `no such table: chunks`.
    """
    settings = settings or get_settings()
    settings.db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(settings.db_path)
    conn.row_factory = sqlite3.Row

    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)

    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    migrate(conn, settings)
    return conn


def migrate(conn: sqlite3.Connection, settings: Settings | None = None) -> None:
    """Apply the static schema, then the dimension-dependent vector table."""
    settings = settings or get_settings()
    conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
    # vec0 needs the dimension baked into the DDL, so it can't live in schema.sql.
    conn.execute(
        f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS chunk_vec USING vec0(
            chunk_id INTEGER PRIMARY KEY,
            embedding FLOAT[{settings.embed_dims}] distance_metric=cosine
        )
        """
    )
    conn.commit()


def index_status(conn: sqlite3.Connection) -> dict[str, object]:
    """Summary of what is currently indexed. Backs `GET /healthz` and the CLI."""
    row = conn.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()
    vec_row = conn.execute("SELECT COUNT(*) AS n FROM chunk_vec").fetchone()
    run = conn.execute(
        "SELECT embed_model, dims, chunk_count, started_at, finished_at"
        " FROM ingest_runs ORDER BY id DESC LIMIT 1"
    ).fetchone()
    book = conn.execute(
        "SELECT slug, title, license, content_hash, ingested_at FROM rulebooks ORDER BY id LIMIT 1"
    ).fetchone()
    return {
        "chunks": row["n"],
        "vectors": vec_row["n"],
        "last_ingest": dict(run) if run else None,
        "rulebook": dict(book) if book else None,
    }


class StaleIndexError(RuntimeError):
    """The index on disk was not built with the embedding model now configured."""


def assert_index_usable(conn: sqlite3.Connection, settings: Settings | None = None) -> None:
    """Fail loudly rather than return confident nonsense.

    Querying a cosine index with vectors from a different embedding model does
    not error — it returns plausible-looking, meaningless neighbours. That is
    the worst failure mode for a citation tool, so check it up front.
    """
    settings = settings or get_settings()
    run = conn.execute(
        "SELECT embed_model, dims FROM ingest_runs WHERE finished_at IS NOT NULL"
        " ORDER BY id DESC LIMIT 1"
    ).fetchone()
    if run is None:
        raise StaleIndexError("No completed ingest found. Run `make ingest` first.")
    if run["embed_model"] != settings.embed_model or run["dims"] != settings.embed_dims:
        raise StaleIndexError(
            f"Index was built with {run['embed_model']} ({run['dims']}d) but the app is "
            f"configured for {settings.embed_model} ({settings.embed_dims}d). "
            "Re-run `make ingest` to rebuild."
        )

"""
RAGSmith – SQLite database layer (stdlib only, no ORM needed)
"""

import sqlite3
import logging
from pathlib import Path

DB_PATH = Path("data/ragsmith.db")
logger = logging.getLogger("ragsmith.db")


def get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db() -> None:
    """Create tables if they don't exist."""
    conn = get_connection()
    try:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS projects (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                name        TEXT    NOT NULL UNIQUE,
                description TEXT    DEFAULT '',
                model       TEXT    DEFAULT 'mistral',
                top_k       INTEGER DEFAULT 5,
                created_at  TEXT    DEFAULT (datetime('now')),
                updated_at  TEXT    DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS documents (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id  INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                filename    TEXT    NOT NULL,
                file_path   TEXT    NOT NULL,
                num_chunks  INTEGER DEFAULT 0,
                status      TEXT    DEFAULT 'pending',
                error_msg   TEXT    DEFAULT NULL,
                created_at  TEXT    DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS query_logs (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id  INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                query_text  TEXT    NOT NULL,
                response    TEXT    NOT NULL,
                num_chunks  INTEGER DEFAULT 0,
                created_at  TEXT    DEFAULT (datetime('now'))
            );
            """
        )
        conn.commit()
        logger.info("Database initialised at %s", DB_PATH)
    finally:
        conn.close()

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

from src.database.schema import CREATE_INDEXES_SQL, CREATE_TABLES_SQL


DEFAULT_DB_PATH = "data/database/stocksense.db"


def get_database_path(db_path: str | None = None) -> Path:
    """
    Return the database path.

    The path can be overridden with:
    - function argument
    - STOCKSENSE_DB_PATH environment variable
    - default local path
    """

    selected_path = db_path or os.getenv("STOCKSENSE_DB_PATH", DEFAULT_DB_PATH)
    return Path(selected_path)


def get_connection(db_path: str | None = None) -> sqlite3.Connection:
    """
    Create and return a SQLite database connection.
    """

    database_path = get_database_path(db_path)

    if str(database_path) != ":memory:":
        database_path.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON;")

    return connection


def initialize_database(db_path: str | None = None) -> None:
    """
    Create database tables and indexes if they do not already exist.
    """

    with get_connection(db_path) as connection:
        connection.executescript(CREATE_TABLES_SQL)
        connection.executescript(CREATE_INDEXES_SQL)
        connection.commit()
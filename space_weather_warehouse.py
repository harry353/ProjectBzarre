import sqlite3
from typing import Iterable, Sequence


class SpaceWeatherWarehouse:
    """
    Lightweight SQLite helper.

    Data sources call ensure_table() with their CREATE TABLE statement and
    insert_rows() with the prepared row tuples. This keeps table-specific
    logic alongside each data-source implementation.
    """

    def __init__(self, db_path="space_weather.db"):
        self.db_path = db_path

    def ensure_table(self, ddl: str):
        """Create the table(s) described by the provided DDL if missing."""
        with sqlite3.connect(self.db_path, timeout=30) as conn:
            conn.execute("PRAGMA busy_timeout=30000")
            conn.executescript(ddl)

    def insert_rows(self, sql: str, rows: Iterable[Sequence]):
        """Bulk insert the given rows using the provided SQL statement."""
        rows = list(rows)
        if not rows:
            return 0

        with sqlite3.connect(self.db_path, timeout=30) as conn:
            conn.execute("PRAGMA busy_timeout=30000")
            conn.executemany(sql, rows)
            conn.commit()
            return len(rows)

from __future__ import annotations

from datetime import date, timedelta
from typing import Iterator, Tuple

from .constants import CHUNK_DAYS


def iter_date_windows(
    start_date: date, end_date: date, chunk_days: int = CHUNK_DAYS
) -> Iterator[Tuple[date, date]]:
    if chunk_days <= 0:
        raise ValueError("chunk_days must be positive.")
    cursor = start_date
    delta = timedelta(days=chunk_days)
    while cursor <= end_date:
        stop = min(cursor + timedelta(days=chunk_days - 1), end_date)
        yield cursor, stop
        cursor += delta


__all__ = ["iter_date_windows"]

from __future__ import annotations

from datetime import datetime

from .constants import LOG_TIME_FORMAT


def stamp(message: str) -> str:
    now = datetime.now().strftime(LOG_TIME_FORMAT)
    return f"[{now}] {message}"


__all__ = ["stamp"]

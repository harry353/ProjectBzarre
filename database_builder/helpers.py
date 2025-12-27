from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Dict, Type
import re

from space_weather_api import SpaceWeatherAPI

def build_source_kwargs(
    cls: Type[SpaceWeatherAPI],
    start_date: date,
    end_date: date,
) -> Dict[str, object]:
    kwargs: Dict[str, object] = {"days": (start_date, end_date)}
    return kwargs


def reset_database(db_path: Path) -> None:
    if db_path.exists():
        db_path.unlink()


def friendly_name(class_name: str) -> str:
    base = re.sub(r"DataSource$", "", class_name)
    tokens = re.findall(r"[A-Z]+(?=[A-Z][a-z]|$)|[A-Z]?[a-z]+|[0-9]+", base)
    label = " ".join(tokens) if tokens else base
    return f"{label} data"


__all__ = ["build_source_kwargs", "reset_database", "friendly_name"]

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing.utils import (
    engineer_hourly_event_features,
    load_event_tables,
    write_sqlite_table,
)

OUTPUT_DB = Path(__file__).resolve().parent / "cme_hourly.db"
TABLE_NAME = "cme_hourly_features"


def main() -> None:
    events = load_event_tables(
        ["lasco_cme_catalog"],
        time_col="time_tag",
        value_cols=[
            "median_velocity",
            "angular_width",
            "min_velocity",
            "max_velocity",
            "halo_class",
        ],
    )
    if events.empty:
        print("[WARN] No CACTUS CME events available.")
        return

    features = engineer_hourly_event_features(
        events,
        time_col="time_tag",
        value_cols=[
            "median_velocity",
            "angular_width",
            "min_velocity",
            "max_velocity",
            "halo_class",
        ],
        prefix="cme_",
    )
    write_sqlite_table(features, OUTPUT_DB, TABLE_NAME)


if __name__ == "__main__":
    main()

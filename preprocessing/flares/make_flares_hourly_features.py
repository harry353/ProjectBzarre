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

OUTPUT_DB = Path(__file__).resolve().parent / "flares_hourly.db"
TABLE_NAME = "flare_hourly_features"


def main() -> None:
    events = load_event_tables(
        ["goes_flares", "goes_flares_archive"],
        time_col="event_time",
        value_cols=[
            "peak_flux_wm2",
            "background_flux",
            "integrated_flux",
            "satellite",
            "flare_class",
            "xrsb_flux",
        ],
        extra_cols=["status"],
    )
    if events.empty:
        print("[WARN] No GOES flare events available.")
        return

    features = engineer_hourly_event_features(
        events,
        time_col="event_time",
        value_cols=["peak_flux_wm2", "background_flux", "integrated_flux"],
        prefix="flare_",
        status_col="status",
    )
    write_sqlite_table(features, OUTPUT_DB, TABLE_NAME)


if __name__ == "__main__":
    main()

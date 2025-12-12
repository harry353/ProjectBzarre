from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing.hourly_utils import (
    read_timeseries_table,
    resample_to_hourly,
    load_hourly_output,
    write_sqlite_table,
    TARGET_CADENCE,
)
import pandas as pd

OUTPUT_DB = Path(__file__).resolve().parent / "solar_wind_hourly.db"
ACE_TABLE = "ace_swepam_hourly"
DSCOVR_TABLE = "dscovr_f1m_hourly"
COMBINED_TABLE = "solar_wind_hourly"
SOLAR_WIND_COLUMNS = ["density", "speed", "temperature"]


def main() -> None:
    ace_df = read_timeseries_table(
        "ace_swepam",
        time_col="time_tag",
        value_cols=SOLAR_WIND_COLUMNS,
        rename_prefix="ace_",
    )
    dscovr_df = read_timeseries_table(
        "dscovr_f1m",
        time_col="time_tag",
        value_cols=SOLAR_WIND_COLUMNS,
        rename_prefix="dscovr_",
    )

    if ace_df.empty and dscovr_df.empty:
        print("[WARN] No solar wind datasets were written.")
        return

    ace_hourly = resample_to_hourly(ace_df, method="mean") if not ace_df.empty else pd.DataFrame()
    dscovr_hourly = resample_to_hourly(dscovr_df, method="mean") if not dscovr_df.empty else pd.DataFrame()

    if not ace_hourly.empty:
        write_sqlite_table(ace_hourly, OUTPUT_DB, ACE_TABLE)
    if not dscovr_hourly.empty:
        write_sqlite_table(dscovr_hourly, OUTPUT_DB, DSCOVR_TABLE)

    combined = build_combined_series(ace_hourly, dscovr_hourly)
    if combined.empty:
        print("[WARN] Combined solar wind dataset is empty.")
        return

    write_sqlite_table(combined, OUTPUT_DB, COMBINED_TABLE)
    print("[OK] Combined solar wind dataset written without gaps.")


def build_combined_series(ace: pd.DataFrame, dscovr: pd.DataFrame) -> pd.DataFrame:
    frames = [df for df in (ace, dscovr) if not df.empty]
    if not frames:
        return pd.DataFrame()

    start = min(df.index.min() for df in frames)
    end = max(df.index.max() for df in frames)
    hourly_index = pd.date_range(
        start=start.floor(TARGET_CADENCE),
        end=end.ceil(TARGET_CADENCE),
        freq=TARGET_CADENCE,
        tz="UTC",
    )
    combined = pd.DataFrame(index=hourly_index)

    for col in SOLAR_WIND_COLUMNS:
        ace_col = f"ace_{col}"
        dscovr_col = f"dscovr_{col}"
        ace_series = ace.get(ace_col)
        dscovr_series = dscovr.get(dscovr_col)
        if ace_series is None and dscovr_series is None:
            combined[col] = pd.Series(index=hourly_index, dtype=float)
            continue
        if ace_series is None:
            merged = dscovr_series.reindex(hourly_index)
        elif dscovr_series is None:
            merged = ace_series.reindex(hourly_index)
        else:
            merged = ace_series.reindex(hourly_index).combine_first(
                dscovr_series.reindex(hourly_index)
            )
        combined[col] = merged

    combined = combined.dropna(how="all")
    return combined


if __name__ == "__main__":
    main()

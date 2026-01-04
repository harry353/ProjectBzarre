from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve()
for parent in PROJECT_ROOT.parents:
    if (parent / "space_weather_api.py").exists():
        PROJECT_ROOT = parent
        break
else:
    PROJECT_ROOT = PROJECT_ROOT.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from pandas.errors import DatabaseError

from preprocessing_pipeline.utils import (
    read_timeseries_table,
    resample_to_hourly,
    write_sqlite_table,
)

# ---------------------------------------------------------------------
# Paths / config
# ---------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

CONFIG = {
    "label": "GOES X-ray flux (XRS)",
    "table": "xray_flux",
    "time_col": "time_tag",
    "value_cols": [
        "irradiance_xrsa1",
        "irradiance_xrsa2",
        "irradiance_xrsb1",
        "irradiance_xrsb2",
        "xrs_ratio",
    ],
    "output": BASE_DIR / "xray_flux_aver.db",
    "method": "mean",
}


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _count_nans(df: pd.DataFrame) -> int:
    return int(df.isna().sum().sum())


def _build_hourly_dataset(
    label: str,
    table: str,
    time_col: str,
    value_cols: Sequence[str],
    output: Path,
    method: str,
) -> pd.DataFrame:
    # --------------------------------------------------------------
    # Load minute-cadence data
    # --------------------------------------------------------------
    try:
        df = read_timeseries_table(
            table,
            time_col=time_col,
            value_cols=value_cols,
        )
    except DatabaseError as exc:
        fallback_cols = ["irradiance_xrsa", "irradiance_xrsb", "xrs_ratio"]
        print("[WARN] Falling back to combined XRS columns:", fallback_cols)
        df = read_timeseries_table(
            table,
            time_col=time_col,
            value_cols=fallback_cols,
        )
        value_cols = fallback_cols

    if df.empty:
        raise RuntimeError(f"No records found in table '{table}'.")

    # --------------------------------------------------------------
    # Replace sentinel missing values ONLY
    # --------------------------------------------------------------
    df[value_cols] = df[value_cols].replace(-9999.0, pd.NA)

    nan_before = _count_nans(df)

    # --------------------------------------------------------------
    # Resample to hourly cadence
    # --------------------------------------------------------------
    hourly = resample_to_hourly(df, method=method)

    nan_after = _count_nans(hourly)

    # --------------------------------------------------------------
    # Persist
    # --------------------------------------------------------------
    write_sqlite_table(hourly, output, "hourly_data")

    print(f"[OK] {label}")
    print(f"     Input rows        : {len(df):,}")
    print(f"     Hourly rows       : {len(hourly):,}")
    print(f"     NaNs before       : {nan_before:,}")
    print(f"     NaNs after        : {nan_after:,}")
    print(f"     Output written to : {output}")

    return hourly


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------
def main() -> None:
    _build_hourly_dataset(
        label=CONFIG["label"],
        table=CONFIG["table"],
        time_col=CONFIG["time_col"],
        value_cols=CONFIG["value_cols"],
        output=CONFIG["output"],
        method=CONFIG["method"],
    )


if __name__ == "__main__":
    main()

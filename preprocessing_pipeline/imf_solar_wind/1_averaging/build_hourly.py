from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

# Resolve absolute path of current script and search for project root (space_weather_api.py)
PROJECT_ROOT = Path(__file__).resolve()
for parent in PROJECT_ROOT.parents:
    if (parent / "space_weather_api.py").exists():
        PROJECT_ROOT = parent
        break
else:
    PROJECT_ROOT = PROJECT_ROOT.parent

# Inject project root into system path to enable module imports from the local codebase
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from preprocessing_pipeline.utils import read_timeseries_table, resample_to_hourly, write_sqlite_table

BASE_DIR = Path(__file__).resolve().parent

# Define separate configurations for ACE and DSCOVR instruments (both SW and IMF)
CONFIGS = [
    {
        "label": "ACE solar wind (SWEPAM)",
        "table": "ace_swepam",
        "time_col": "time_tag",
        "value_cols": ["density", "speed", "temperature"],
        "output": BASE_DIR / "ace_swepam_aver.db",
        "method": "mean", # Use arithmetic mean for scalar SW physical parameters
    },
    {
        "label": "DSCOVR solar wind (F1M)",
        "table": "dscovr_f1m",
        "time_col": "time_tag",
        "value_cols": ["density", "speed", "temperature"],
        "output": BASE_DIR / "dscovr_f1m_aver.db",
        "method": "mean",
    },
    {
        "label": "ACE IMF (MFI)",
        "table": "ace_mfi",
        "time_col": "time_tag",
        "value_cols": ["bx_gsm", "by_gsm", "bz_gsm", "bt"],
        "output": BASE_DIR / "ace_mfi_aver.db",
        "method": "mean", # Hourly average of magnetic field vector components
    },
    {
        "label": "DSCOVR IMF (M1M)",
        "table": "dscovr_m1m",
        "time_col": "time_tag",
        "value_cols": ["bt", "bx", "by", "bz"],
        "output": BASE_DIR / "dscovr_m1m_aver.db",
        "method": "mean",
    },
]


# Loads raw sensor data, aggregates to hourly bins, and persists result to a staging DB
def _build_hourly_dataset(
    label: str,
    table: str,
    time_col: str,
    value_cols: Sequence[str],
    output: Path,
    method: str,
) -> pd.DataFrame:
    # Query raw time-series from central telemetry database
    df = read_timeseries_table(
        table,
        time_col=time_col,
        value_cols=value_cols,
    )
    if df.empty:
        raise RuntimeError(f"No records found in table '{table}'.")
    
    # Perform temporal resampling to align all sensors to a uniform hourly grid
    hourly = resample_to_hourly(df, method=method)
    
    # Save processed hourly data to a local SQLite database for downstream stages
    write_sqlite_table(hourly, output, "hourly_data")
    print(f"[OK] {label} hourly dataset written to {output}")
    return hourly


def main() -> None:
    # Process all instrument configurations sequentially
    for cfg in CONFIGS:
        _build_hourly_dataset(
            label=cfg["label"],
            table=cfg["table"],
            time_col=cfg["time_col"],
            value_cols=cfg["value_cols"],
            output=cfg["output"],
            method=cfg["method"],
        )


if __name__ == "__main__":
    main()

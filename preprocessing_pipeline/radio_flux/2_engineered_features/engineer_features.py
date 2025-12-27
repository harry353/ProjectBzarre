from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve()
for parent in PROJECT_ROOT.parents:
    if (parent / "space_weather_api.py").exists():
        PROJECT_ROOT = parent
        break
else:
    PROJECT_ROOT = PROJECT_ROOT.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from preprocessing_pipeline.utils import load_hourly_output, write_sqlite_table

STAGE_DIR = Path(__file__).resolve().parent
FILTERED_DB = (
    STAGE_DIR.parents[1] / "radio_flux" / "1_hard_filtering" / "radio_flux_filt.db"
)
FILTERED_TABLE = "filtered_data"
OUTPUT_DB = STAGE_DIR / "radio_flux_filt_eng.db"
OUTPUT_TABLE = "engineered_features"


def _add_f107_features(df: pd.DataFrame) -> pd.DataFrame:
    working = df.copy()
    if "adjusted_flux" not in working.columns:
        raise RuntimeError("Expected 'adjusted_flux' column in filtered dataset.")

    working = working.rename(columns={"adjusted_flux": "f107"})

    # Log-transform (safe for non-positive values)
    valid = working["f107"] > 0
    working["log_f107"] = pd.Series(np.nan, index=working.index)
    working.loc[valid, "log_f107"] = np.log(working.loc[valid, "f107"])

    for lag, days in [("f107_lag_1d", 1), ("f107_lag_7d", 7), ("f107_lag_27d", 27)]:
        working[lag] = working["f107"].shift(days)

    working["f107_mean_7d"] = working["f107"].rolling(window=7, min_periods=1).mean()
    working["f107_mean_27d"] = working["f107"].rolling(window=27, min_periods=1).mean()
    working["f107_mean_81d"] = working["f107"].rolling(window=81, min_periods=1).mean()
    working["f107_std_27d"] = working["f107"].rolling(window=27, min_periods=1).std()

    working["f107_slope_27d"] = (
        working["f107"] - working["f107"].shift(27)
    ) / 27.0

    regime = pd.cut(
        working["f107"],
        bins=[-np.inf, 80, 150, np.inf],
        labels=[0, 1, 2],
        right=False,
    )
    working["f107_regime"] = regime.astype("Int64")

    # Ensure no infinite values remain
    working = working.replace([np.inf, -np.inf], np.nan)
    return working


def engineer_radio_flux_features() -> pd.DataFrame:
    df = load_hourly_output(FILTERED_DB, FILTERED_TABLE)
    if df.empty:
        raise RuntimeError("Filtered radio flux dataset not found; run filtering first.")

    features = _add_f107_features(df)
    write_sqlite_table(features, OUTPUT_DB, OUTPUT_TABLE)
    print(f"[OK] Radio flux engineered features saved to {OUTPUT_DB}")
    return features


def main() -> None:
    engineer_radio_flux_features()


if __name__ == "__main__":
    main()

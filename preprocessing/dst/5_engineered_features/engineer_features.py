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

import pandas as pd

from preprocessing.utils import load_hourly_output, write_sqlite_table

STAGE_DIR = Path(__file__).resolve().parent
IMPUTED_DB = STAGE_DIR.parents[1] / "dst" / "4_imputation" / "dst_imp.db"
IMPUTED_TABLE = "imputed_data"
OUTPUT_DB = STAGE_DIR / "dst_eng.db"
OUTPUT_TABLE = "engineered_features"


def _add_dst_features(df: pd.DataFrame) -> pd.DataFrame:
    working = df.copy()
    if "dst" not in working.columns:
        raise RuntimeError("DST column missing from imputed dataset.")

    for lag in range(1, 13):
        working[f"dst_lag_{lag}"] = working["dst"].shift(lag)

    working["dst_mean_6h"] = working["dst"].rolling(window=6, min_periods=1).mean()
    working["dst_std_6h"] = working["dst"].rolling(window=6, min_periods=1).std()
    working["dst_derivative"] = working["dst"] - working["dst"].shift(1)

    lag_columns = [col for col in working.columns if "lag_" in col]
    if lag_columns:
        working = working.dropna(subset=lag_columns)
    return working


def engineer_dst_features() -> pd.DataFrame:
    df = load_hourly_output(IMPUTED_DB, IMPUTED_TABLE)
    if df.empty:
        raise RuntimeError("Imputed DST dataset not found; run imputation first.")
    features = _add_dst_features(df)
    write_sqlite_table(features, OUTPUT_DB, OUTPUT_TABLE)
    print(f"[OK] DST engineered features saved to {OUTPUT_DB}")
    return features


def main() -> None:
    engineer_dst_features()


if __name__ == "__main__":
    main()

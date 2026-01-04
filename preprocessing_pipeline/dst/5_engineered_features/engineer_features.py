from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve()
for parent in PROJECT_ROOT.parents:
    if (parent / "space_weather_api.py").exists():
        PROJECT_ROOT = parent
        break
else:
    PROJECT_ROOT = PROJECT_ROOT.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing_pipeline.utils import load_hourly_output, write_sqlite_table

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
STAGE_DIR = Path(__file__).resolve().parent
IMPUTED_DB = (
    STAGE_DIR.parents[1]
    / "dst"
    / "4_imputation"
    / "dst_aver_filt_imp.db"
)
IMPUTED_TABLE = "imputed_data"

OUTPUT_DB = STAGE_DIR / "dst_aver_filt_imp_eng.db"
OUTPUT_TABLE = "engineered_features"

# ---------------------------------------------------------------------
# Feature engineering (HOURLY, NO AGGREGATES)
# ---------------------------------------------------------------------
def _add_dst_features(df: pd.DataFrame) -> pd.DataFrame:
    working = df.copy()

    if "dst" not in working.columns:
        raise RuntimeError("DST column missing from imputed dataset.")

    dst = working["dst"].astype(float)

    # --------------------------------------------------------------
    # Core DST features (6 total)
    # --------------------------------------------------------------
    working["dst"] = dst

    working["dst_delta_1h"] = dst.diff(1)
    working["dst_delta_3h"] = dst.diff(3)
    working["dst_delta_6h"] = dst.diff(6)

    working["dst_negative_flag"] = (dst < 0).astype(int)

    working["dst_recovery_flag"] = (
        (dst < -30) & (dst.diff(1) > 0)
    ).astype(int)

    # --------------------------------------------------------------
    # Cleanup
    # --------------------------------------------------------------
    working = working.dropna()

    return working

# ---------------------------------------------------------------------
# Pipeline entry
# ---------------------------------------------------------------------
def engineer_dst_features() -> pd.DataFrame:
    df = load_hourly_output(IMPUTED_DB, IMPUTED_TABLE)
    if df.empty:
        raise RuntimeError("Imputed DST dataset not found; run imputation first.")

    features = _add_dst_features(df)

    write_sqlite_table(features, OUTPUT_DB, OUTPUT_TABLE)

    print(f"[OK] DST engineered features saved to {OUTPUT_DB}")
    print(f"Rows written: {len(features):,}")

    return features

# ---------------------------------------------------------------------
def main() -> None:
    engineer_dst_features()

if __name__ == "__main__":
    main()


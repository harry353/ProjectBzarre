from __future__ import annotations

import sqlite3
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve()
for p in PROJECT_ROOT.parents:
    if (p / "space_weather_api.py").exists():
        PROJECT_ROOT = p
        break

PIPELINE_ROOT = PROJECT_ROOT / "preprocessing_pipeline"

IMF_DB = (
    PIPELINE_ROOT
    / "imf_solar_wind"
    / "6_engineered_features"
    / "imf_solar_wind_aver_comb_filt_imp_eng.db"
)

MODEL_DIR = PIPELINE_ROOT / "features_targets" / "disturbed_label"
MODEL_PATH = MODEL_DIR / "stage1_model.joblib"
SCALER_PATH = MODEL_DIR / "stage1_scaler.joblib"

OUT_DB = MODEL_DIR / "stage1_probs.db"
OUT_TABLE = "stage1_probs"

# ------------------------------------------------------------------
# Loaders
# ------------------------------------------------------------------

def load_imf() -> pd.DataFrame:
    with sqlite3.connect(IMF_DB) as conn:
        df = pd.read_sql(
            """
            SELECT time_tag, bz_gse, by_gse, bt,
                   speed, dynamic_pressure, newell_dphi_dt
            FROM engineered_features
            """,
            conn,
            parse_dates=["time_tag"],
        )
    df = df.set_index("time_tag").sort_index()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df

# ------------------------------------------------------------------
# Feature engineering (MUST match Stage-1 training)
# ------------------------------------------------------------------

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)

    X["bz"] = df["bz_gse"]
    X["bt"] = df["bt"]
    X["speed"] = df["speed"]
    X["pressure"] = df["dynamic_pressure"]
    X["newell"] = df["newell_dphi_dt"]

    X["clock_angle"] = np.degrees(
        np.arctan2(df["by_gse"].abs(), df["bz_gse"])
    )

    ey = df["speed"] * (-df["bz_gse"].clip(upper=0))
    X["ey"] = ey

    X["bz_mean_3h"] = df["bz_gse"].rolling(3).mean()
    X["ey_mean_3h"] = ey.rolling(3).mean()
    X["newell_sum_3h"] = df["newell_dphi_dt"].rolling(3).sum()

    return X.dropna()

# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main() -> None:
    print("[INFO] Loading IMF data")
    imf = load_imf()

    print("[INFO] Building features")
    X = build_features(imf)

    print("[INFO] Loading Stage-1 model")
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    Xs = scaler.transform(X)

    print("[INFO] Computing probabilities")
    probs = model.predict_proba(Xs)[:, 1]

    out = pd.DataFrame(
        {
            "time_tag": X.index,
            "p_stage1": probs,
        }
    )

    with sqlite3.connect(OUT_DB) as conn:
        out.to_sql(OUT_TABLE, conn, if_exists="replace", index=False)

    print("[OK] Stage-1 probabilities written")
    print(f"     Database : {OUT_DB}")
    print(f"     Table    : {OUT_TABLE}")
    print(f"     Rows     : {len(out):,}")
    print(f"     Mean p   : {out['p_stage1'].mean():.3f}")

if __name__ == "__main__":
    main()

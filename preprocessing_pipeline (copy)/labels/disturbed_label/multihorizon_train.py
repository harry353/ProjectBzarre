from __future__ import annotations

import sqlite3
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score
from sklearn.preprocessing import StandardScaler

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

SSC_DB = PIPELINE_ROOT / "features_targets" / "full_storm_label" / "full_storm_labels.db"

OUT_DIR = Path(__file__).resolve().parent / "multi_horizon_models"
OUT_DIR.mkdir(exist_ok=True)

# ------------------------------------------------------------------
# Parameters
# ------------------------------------------------------------------

HORIZONS = range(1, 9)   # 1h ... 8h
PROB_THRESHOLD = 0.1

TRAIN_START = "1999-01-01"
TRAIN_END   = "2016-12-31"

TEST_START  = "2021-01-01"
TEST_END    = "2025-11-30"

# ------------------------------------------------------------------
# Loaders
# ------------------------------------------------------------------

def ensure_utc_index(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    if idx.tz is None:
        return idx.tz_localize("UTC")
    return idx.tz_convert("UTC")


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
    df.index = ensure_utc_index(df.index)
    return df


def load_dst() -> pd.DataFrame:
    dst_db = PIPELINE_ROOT / "dst" / "1_averaging" / "dst_aver.db"
    with sqlite3.connect(dst_db) as conn:
        df = pd.read_sql(
            "SELECT time_tag AS t, dst FROM hourly_data",
            conn,
            parse_dates=["t"],
        )
    df = df.set_index("t").sort_index()
    df.index = ensure_utc_index(df.index)
    return df


def load_kp() -> pd.DataFrame:
    kp_db = PIPELINE_ROOT / "kp_index" / "1_averaging" / "kp_index_aver.db"
    with sqlite3.connect(kp_db) as conn:
        df = pd.read_sql(
            "SELECT time_tag AS t, kp_index FROM hourly_data",
            conn,
            parse_dates=["t"],
        )
    df = df.set_index("t").sort_index()
    df.index = ensure_utc_index(df.index)
    return df


def load_storm_onsets() -> pd.Series:
    frames = []
    with sqlite3.connect(SSC_DB) as conn:
        for t in ["storm_full_storm_train", "storm_full_storm_validation", "storm_full_storm_test"]:
            if not pd.read_sql(
                "SELECT name FROM sqlite_master WHERE name=?",
                conn,
                params=(t,),
            ).empty:
                frames.append(
                    pd.read_sql(
                        f"SELECT timestamp, storm_flag FROM {t}",
                        conn,
                        parse_dates=["timestamp"],
                    )
                )

    full_storm = pd.concat(frames).drop_duplicates("timestamp")
    full_storm = full_storm.set_index("timestamp").sort_index()
    return full_storm["storm_flag"]

# ------------------------------------------------------------------
# Feature engineering
# ------------------------------------------------------------------

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)

    # ------------------------------------------------------------------
    # Solar wind / IMF (existing)
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Ground indices (raw, past-only)
    # ------------------------------------------------------------------

    X["dst"] = df["dst"]
    X["kp"] = df["kp"]

    # ------------------------------------------------------------------
    # Engineered Dst features (past-only)
    # ------------------------------------------------------------------

    # Ring current strength
    X["dst_min_6h"] = df["dst"].rolling(6).min()

    # Trend: negative = intensifying storm, positive = recovery
    X["dst_slope_6h"] = df["dst"].diff().rolling(6).mean()

    # ------------------------------------------------------------------
    # Engineered Kp features (past-only)
    # ------------------------------------------------------------------

    # Recent global disturbance level
    X["kp_max_6h"] = df["kp"].rolling(6).max()

    # Persistence of elevated activity
    X["kp_ge4_frac_6h"] = (df["kp"] >= 4).rolling(6).mean()

    # ------------------------------------------------------------------
    # Final cleanup
    # ------------------------------------------------------------------

    return X.dropna()

# ------------------------------------------------------------------
# Target construction (parametric horizon)
# ------------------------------------------------------------------

def build_target(
    index: pd.DatetimeIndex,
    storm_onsets: pd.Series,
    lead_hours: int,
) -> pd.Series:
    y = pd.Series(0, index=index)
    storm_times = storm_onsets[storm_onsets == 1].index

    for t in index:
        if ((storm_times > t) &
            (storm_times <= t + pd.Timedelta(hours=lead_hours))).any():
            y.loc[t] = 1

    return y

# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main() -> None:
    imf = load_imf()
    dst = load_dst()
    kp = load_kp()
    storms = load_storm_onsets()

    combined = (
        imf.join(dst.rename(columns={"dst": "dst"}), how="inner")
        .join(kp.rename(columns={"kp_index": "kp"}), how="inner")
    )
    X = build_features(combined)

    for h in HORIZONS:
        print(f"\n[INFO] Training horizon = {h}h")

        y = build_target(X.index, storms, lead_hours=h)
        data = X.join(y.rename("target")).dropna()

        train = data.loc[TRAIN_START:TRAIN_END]
        test  = data.loc[TEST_START:TEST_END]

        X_train, y_train = train.drop(columns="target"), train["target"]
        X_test, y_test   = test.drop(columns="target"), test["target"]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)

        clf = LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            n_jobs=-1,
        )

        clf.fit(X_train, y_train)

        probs = clf.predict_proba(X_test)[:, 1]
        y_pred = probs >= PROB_THRESHOLD

        recall = recall_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)

        print(f"  recall    = {recall:.3f}")
        print(f"  precision = {precision:.3f}")

        model_path = OUT_DIR / f"stage1_model_{h}h.joblib"
        scaler_path = OUT_DIR / f"stage1_scaler_{h}h.joblib"

        joblib.dump(clf, model_path)
        joblib.dump(scaler, scaler_path)

        print(f"  saved model  → {model_path.name}")
        print(f"  saved scaler → {scaler_path.name}")

# ------------------------------------------------------------------

if __name__ == "__main__":
    main()

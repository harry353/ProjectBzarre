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

MODEL_PATH = Path(__file__).resolve().parent / "stage1_model.joblib"
SCALER_PATH = Path(__file__).resolve().parent / "stage1_scaler.joblib"

# ------------------------------------------------------------------
# Parameters
# ------------------------------------------------------------------

LEAD_HOURS = 8
PROB_THRESHOLD = 0.1

TRAIN_START = "1999-01-01"
TRAIN_END   = "2016-12-31"

TEST_START  = "2021-01-01"
TEST_END    = "2025-11-30"

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
    return df.set_index("time_tag").sort_index()


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
# Target construction
# ------------------------------------------------------------------

def build_target(index: pd.DatetimeIndex, storm_onsets: pd.Series) -> pd.Series:
    y = pd.Series(0, index=index)
    storm_times = storm_onsets[storm_onsets == 1].index

    for t in index:
        if ((storm_times > t) &
            (storm_times <= t + pd.Timedelta(hours=LEAD_HOURS))).any():
            y.loc[t] = 1

    return y

# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main() -> None:
    imf = load_imf()
    storms = load_storm_onsets()

    X = build_features(imf)
    y = build_target(X.index, storms)

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

    print(f"Stage-1 classifier (12h lead, p ≥ {PROB_THRESHOLD}):")
    print(f"  recall    = {recall:.3f}")
    print(f"  precision = {precision:.3f}")

    joblib.dump(clf, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    print(f"[OK] Saved model  → {MODEL_PATH}")
    print(f"[OK] Saved scaler → {SCALER_PATH}")

    print("\nTop coefficients:")
    for name, coef in sorted(
        zip(X_train.shape[1] * [""] + train.drop(columns="target").columns, clf.coef_[0]),
        key=lambda x: abs(x[1]),
        reverse=True,
    )[:10]:
        print(f"  {name:15s} {coef:+.3f}")


if __name__ == "__main__":
    main()

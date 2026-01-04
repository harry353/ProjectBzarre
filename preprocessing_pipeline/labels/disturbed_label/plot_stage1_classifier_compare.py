from __future__ import annotations

import sqlite3
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm, colors

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

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

DST_DB = PIPELINE_ROOT / "dst" / "1_averaging" / "dst_aver.db"

LOG_MODEL_PATH = (
    PIPELINE_ROOT / "features_targets" / "disturbed_label" / "stage1_model.joblib"
)
LOG_SCALER_PATH = (
    PIPELINE_ROOT / "features_targets" / "disturbed_label" / "stage1_scaler.joblib"
)

XGB_MODEL_PATH = (
    PIPELINE_ROOT / "features_targets" / "disturbed_label" / "stage1_xgb_model.joblib"
)
XGB_FEATURES_PATH = (
    PIPELINE_ROOT / "features_targets" / "disturbed_label" / "stage1_xgb_features.joblib"
)

# ---------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------

YEAR = 2024
LOG_THRESHOLD = 0.35
XGB_THRESHOLD = 0.35
PLOT_ONLY_ABOVE_THRESH = False
PLOT_HOURS = pd.Timedelta(hours=1)

# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def ensure_utc_index(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    if idx.tz is None:
        return idx.tz_localize("UTC")
    return idx.tz_convert("UTC")

# ---------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------

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
    with sqlite3.connect(DST_DB) as conn:
        df = pd.read_sql(
            "SELECT time_tag AS t, dst FROM hourly_data",
            conn,
            parse_dates=["t"],
        )
    df = df.set_index("t").sort_index()
    df.index = ensure_utc_index(df.index)
    return df

# ---------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------

def build_features_log(df: pd.DataFrame) -> pd.DataFrame:
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


def build_features_xgb(df: pd.DataFrame) -> pd.DataFrame:
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

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    imf = load_imf()
    dst = load_dst()

    start = pd.Timestamp(f"{YEAR}-01-01", tz="UTC")
    end = start + pd.DateOffset(years=1)

    dst = dst.loc[start:end]

    log_X = build_features_log(imf)
    log_scaler = joblib.load(LOG_SCALER_PATH)
    log_model = joblib.load(LOG_MODEL_PATH)
    log_Xs = log_scaler.transform(log_X)
    log_probs = pd.Series(
        log_model.predict_proba(log_Xs)[:, 1],
        index=log_X.index,
        name="p",
    ).loc[start:end]

    xgb_X = build_features_xgb(imf)
    xgb_model = joblib.load(XGB_MODEL_PATH)
    xgb_features = joblib.load(XGB_FEATURES_PATH)
    xgb_X = xgb_X.reindex(columns=xgb_features)
    xgb_probs = pd.Series(
        xgb_model.predict_proba(xgb_X)[:, 1],
        index=xgb_X.index,
        name="p",
    ).loc[start:end]

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 6), sharex=True, height_ratios=[1, 1]
    )

    ax1.plot(dst.index, dst["dst"], color="tab:blue", lw=1)
    ax2.plot(dst.index, dst["dst"], color="tab:blue", lw=1)

    norm = colors.Normalize(vmin=0.0, vmax=1.0)
    cmap = cm.get_cmap("YlOrRd")

    for t, p in log_probs.items():
        if PLOT_ONLY_ABOVE_THRESH and p < LOG_THRESHOLD:
            continue
        color = cmap(norm(p))
        ax1.axvspan(t, t + PLOT_HOURS, color=color, alpha=0.35, lw=0)

    for t, p in xgb_probs.items():
        if PLOT_ONLY_ABOVE_THRESH and p < XGB_THRESHOLD:
            continue
        color = cmap(norm(p))
        ax2.axvspan(t, t + PLOT_HOURS, color=color, alpha=0.35, lw=0)

    ax1.axhline(-50, ls=":", color="black", alpha=0.4)
    ax1.axhline(0, ls=":", color="black", alpha=0.3)
    ax2.axhline(-50, ls=":", color="black", alpha=0.4)
    ax2.axhline(0, ls=":", color="black", alpha=0.3)

    ax1.set_ylabel("Dst (nT)")
    ax2.set_ylabel("Dst (nT)")
    ax2.set_xlabel("Time")

    ax1.set_title(f"Logistic regression probabilities (p ≥ {LOG_THRESHOLD}) – {YEAR}")
    ax2.set_title(f"XGBoost probabilities (p ≥ {XGB_THRESHOLD}) – {YEAR}")

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig.colorbar(
        sm,
        ax=[ax1, ax2],
        location="right",
        pad=0.02,
        label="Stage-1 probability",
    )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

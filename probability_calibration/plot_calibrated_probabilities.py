from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm, colors
from xgboost import XGBClassifier


# ---------------------------------------------------------------------
# Project resolution
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve()
for parent in PROJECT_ROOT.parents:
    if (parent / "space_weather_api.py").exists():
        PROJECT_ROOT = parent
        break
else:
    PROJECT_ROOT = PROJECT_ROOT.parent


PIPELINE_ROOT = PROJECT_ROOT / "preprocessing_pipeline"
MERGED_DB = PIPELINE_ROOT / "check_multicolinearity" / "all_preprocessed_sources.db"
DST_DB = PIPELINE_ROOT / "dst" / "1_averaging" / "dst_aver.db"

HORIZON_H = 1
MODEL_DIR = PROJECT_ROOT / "modeling_pipeline" / f"output_h{HORIZON_H}"
MODEL_PATH = MODEL_DIR / "daily_storm_model.json"
FEATURES_JSON = MODEL_DIR / "daily_storm_features.json"
CALIBRATION_DIR = PROJECT_ROOT / "probability_calibration" / f"calibration_h{HORIZON_H}"
CALIBRATION_METADATA = CALIBRATION_DIR / "calibration_metadata.json"

YEAR = 2024
PLOT_ONLY_ABOVE = False
PLOT_THRESHOLD = 0.5
WINDOW_START = None # pd.Timestamp(f"{YEAR}-05-10", tz="UTC")
WINDOW_END = None   # pd.Timestamp(f"{YEAR}-05-13", tz="UTC")


# ---------------------------------------------------------------------
# Solar cycle phase (simple linear proxy)
# ---------------------------------------------------------------------
CYCLE_MINIMA = [
    pd.Timestamp("1996-08-01", tz="UTC"),
    pd.Timestamp("2008-12-01", tz="UTC"),
    pd.Timestamp("2019-12-01", tz="UTC"),
    pd.Timestamp("2031-01-01", tz="UTC"),
]


def solar_cycle_phase(ts: pd.Timestamp) -> float:
    for start, end in zip(CYCLE_MINIMA[:-1], CYCLE_MINIMA[1:]):
        if start <= ts < end:
            return float((ts - start) / (end - start))
    return 0.5


# ---------------------------------------------------------------------
# Regime-aware calibration
# ---------------------------------------------------------------------
with open(CALIBRATION_METADATA, "r", encoding="utf-8") as fp:
    CAL_META = json.load(fp)

REGIME_BINS = CAL_META["regime_bins"]
CALIBRATORS = {
    name: joblib.load(CALIBRATION_DIR / fname)
    for name, fname in CAL_META["calibrators"].items()
}


def assign_regime(phase: float) -> str:
    for name, (lo, hi) in REGIME_BINS.items():
        if lo <= phase < hi or (name == "declining" and phase == 1.0):
            return name
    raise ValueError("Invalid solar cycle phase")


def calibrate_probability(raw_prob: float, phase: float) -> float:
    p = min(max(float(raw_prob), 0.0), 1.0)
    regime = assign_regime(phase)
    return float(CALIBRATORS[regime].predict([p])[0])


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _ensure_utc(series: pd.Series) -> pd.Series:
    ts = pd.to_datetime(series, errors="coerce")
    return ts.dt.tz_localize("UTC") if ts.dt.tz is None else ts.dt.tz_convert("UTC")


def _load_dst() -> pd.Series:
    with sqlite3.connect(DST_DB) as conn:
        df = pd.read_sql(
            "SELECT time_tag AS t, dst FROM hourly_data",
            conn,
            parse_dates=["t"],
        )
    df = df.set_index("t").sort_index()
    df.index = _ensure_utc(df.index.to_series())
    return df["dst"]


def _load_feature_order() -> list[str]:
    with open(FEATURES_JSON, "r", encoding="utf-8") as fp:
        payload = json.load(fp)
    return payload["feature_order"]


def _load_merged_year(year: int, feature_cols: list[str]) -> pd.DataFrame:
    frames = []
    with sqlite3.connect(MERGED_DB) as conn:
        for split in ("train", "validation", "test"):
            df = pd.read_sql_query(f"SELECT * FROM merged_{split}", conn)
            df["timestamp"] = _ensure_utc(df["timestamp"])
            frames.append(df)

    merged = (
        pd.concat(frames)
        .drop_duplicates("timestamp")
        .sort_values("timestamp")
        .set_index("timestamp")
    )

    start = pd.Timestamp(year=year, month=1, day=1, tz="UTC")
    end = start + pd.DateOffset(years=1)
    merged = merged.loc[start:end]

    X = merged[feature_cols].fillna(0.0).to_numpy(np.float32)
    merged["_X"] = list(X)
    return merged


def _load_model() -> XGBClassifier:
    model = XGBClassifier()
    model.load_model(MODEL_PATH)
    return model


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main() -> None:
    feature_cols = _load_feature_order()
    merged = _load_merged_year(YEAR, feature_cols)
    dst = _load_dst()
    model = _load_model()

    X = np.vstack(merged["_X"].to_numpy())
    raw_probs = model.predict_proba(X)[:, 1]

    phases = merged.index.map(solar_cycle_phase)
    calibrated = [
        calibrate_probability(p, ph) for p, ph in zip(raw_probs, phases)
    ]

    probs = pd.Series(calibrated, index=merged.index)

    dst = dst.loc[probs.index.min():probs.index.max()]
    probs = probs.reindex(dst.index)
    mask = probs.notna()
    probs = probs.loc[mask]
    dst = dst.loc[mask]
    dst = dst.loc[WINDOW_START:WINDOW_END]
    probs = probs.loc[WINDOW_START:WINDOW_END]

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(dst.index, dst.values, color="tab:blue", lw=1)

    norm = colors.Normalize(0, 1)
    cmap = cm.get_cmap("YlOrRd")

    for t, p in probs.items():
        if PLOT_ONLY_ABOVE and p < PLOT_THRESHOLD:
            continue
        ax.axvspan(t, t + pd.Timedelta(hours=1), color=cmap(norm(p)), alpha=0.15)

    ax.axhline(-50, ls=":", color="black", alpha=0.4)
    ax.set_title(f"Dst with calibrated storm probability (t+{HORIZON_H}h) – {YEAR}")
    ax.set_ylabel("Dst (nT)")

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Calibrated P(storm)")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

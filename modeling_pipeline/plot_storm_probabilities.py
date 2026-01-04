from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm, colors
from xgboost import XGBClassifier


PROJECT_ROOT = Path(__file__).resolve()
for parent in PROJECT_ROOT.parents:
    if (parent / "space_weather_api.py").exists():
        PROJECT_ROOT = parent
        break
else:
    PROJECT_ROOT = PROJECT_ROOT.parent

PIPELINE_ROOT = PROJECT_ROOT / "preprocessing_pipeline"
MERGED_DB = PIPELINE_ROOT / "final" / "all_preprocessed_sources.db"
DST_DB = PIPELINE_ROOT / "dst" / "1_averaging" / "dst_aver.db"

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
MODEL_PATH = OUTPUT_DIR / "daily_storm_model.json"
FEATURES_JSON = OUTPUT_DIR / "daily_storm_features.json"

YEAR = 2024
PLOT_ONLY_ABOVE = False
PLOT_THRESHOLD = 0.5


def _ensure_utc_index(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    if idx.tz is None:
        return idx.tz_localize("UTC")
    return idx.tz_convert("UTC")


def _ensure_utc(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce")
    if parsed.dt.tz is None:
        return parsed.dt.tz_localize("UTC")
    return parsed.dt.tz_convert("UTC")


def _load_dst() -> pd.Series:
    with sqlite3.connect(DST_DB) as conn:
        df = pd.read_sql(
            "SELECT time_tag AS t, dst FROM hourly_data",
            conn,
            parse_dates=["t"],
        )
    df = df.set_index("t").sort_index()
    df.index = _ensure_utc_index(df.index)
    return df["dst"]


def _load_feature_order() -> list[str]:
    if not FEATURES_JSON.exists():
        raise FileNotFoundError(f"Missing feature contract: {FEATURES_JSON}")
    with open(FEATURES_JSON, "r", encoding="utf-8") as fp:
        payload = json.load(fp)
    order = payload.get("feature_order", [])
    if not order:
        raise RuntimeError("Feature contract missing 'feature_order'.")
    return order


def _load_merged_year(year: int, feature_cols: list[str]) -> pd.DataFrame:
    frames = []
    with sqlite3.connect(MERGED_DB) as conn:
        for split in ("train", "validation", "test"):
            table = f"merged_{split}"
            df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
            if df.empty:
                continue
            if "timestamp" not in df.columns:
                raise RuntimeError(f"Missing timestamp column in {table}.")
            df["timestamp"] = _ensure_utc(df["timestamp"])
            df = df.dropna(subset=["timestamp"])
            frames.append(df)

    if not frames:
        raise RuntimeError("No merged data found to plot.")

    merged = pd.concat(frames, ignore_index=True).drop_duplicates("timestamp")
    merged = merged.sort_values("timestamp")

    start = pd.Timestamp(year=year, month=1, day=1, tz="UTC")
    end = start + pd.DateOffset(years=1)
    merged = merged.loc[(merged["timestamp"] >= start) & (merged["timestamp"] < end)]
    if merged.empty:
        raise RuntimeError(f"No merged rows found for year {year}.")

    merged = merged.set_index("timestamp")
    X = merged.reindex(columns=feature_cols).fillna(0.0).to_numpy(dtype=np.float32)
    merged["_model_matrix"] = list(X)
    return merged


def _load_model() -> XGBClassifier:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model: {MODEL_PATH}")
    model = XGBClassifier()
    model.load_model(MODEL_PATH)
    return model


def main() -> None:
    feature_cols = _load_feature_order()
    merged = _load_merged_year(YEAR, feature_cols)
    dst = _load_dst()
    model = _load_model()

    X = np.vstack(merged["_model_matrix"].to_numpy())
    probs = pd.Series(model.predict_proba(X)[:, 1], index=merged.index, name="p")

    start = pd.Timestamp(f"{YEAR}-01-01", tz="UTC")
    end = start + pd.DateOffset(years=1)
    dst_slice = dst.loc[(dst.index >= start) & (dst.index < end)]
    probs = probs.loc[(probs.index >= start) & (probs.index < end)]

    if dst_slice.empty or probs.empty:
        raise RuntimeError("No overlapping Dst/probability data to plot.")

    fig, ax = plt.subplots(1, 1, figsize=(14, 4))
    ax.plot(dst_slice.index, dst_slice.values, color="tab:blue", lw=1, label="Dst")

    norm = colors.Normalize(vmin=0.0, vmax=1.0)
    cmap = cm.get_cmap("YlOrRd")
    for t, p in probs.items():
        if PLOT_ONLY_ABOVE and p < PLOT_THRESHOLD:
            continue
        ax.axvspan(t, t + pd.Timedelta(hours=1), color=cmap(norm(p)), alpha=0.35, lw=0)

    ax.axhline(-50, ls=":", color="black", alpha=0.4)
    ax.axhline(0, ls=":", color="black", alpha=0.3)
    ax.set_ylabel("Dst (nT)")
    ax.set_xlabel("Time")
    ax.set_title(f"Dst with storm probability (t+4h) – {YEAR}")

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, pad=0.02, label="P(storm in 4h)")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

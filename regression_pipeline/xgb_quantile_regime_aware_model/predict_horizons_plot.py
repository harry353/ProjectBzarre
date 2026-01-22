from __future__ import annotations

import sqlite3
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURES_DB = PROJECT_ROOT / "regression_pipeline" / "pca_features.db"
LABELS_DB = PROJECT_ROOT / "preprocessing_pipeline" / "labels" / "dst_regression" / "dst_regression_labels.db"
DST_DB = PROJECT_ROOT / "preprocessing_pipeline" / "check_multicolinearity" / "all_preprocessed_sources.db"

FEATURE_TABLES = {
    "train": "pca_train",
    "validation": "pca_validation",
    "test": "pca_test",
}
LABEL_TABLES = {
    "train": "dst_regression_train",
    "validation": "dst_regression_validation",
    "test": "dst_regression_test",
}
DST_TABLES = {
    "train": "merged_train",
    "validation": "merged_validation",
    "test": "merged_test",
}

MODEL_BASE = PROJECT_ROOT / "regression_pipeline" / "xgb_quantile_regime_aware_model"
HORIZONS = range(1, 2)
QUANTILES = [0.1, 0.5, 0.9]
YEAR = 2024  # edit as needed
DST_THRESHOLD = -20.0
REGIME = "both"  # "storm", "calm", or "both"


def _load_table(db_path: Path, table: str) -> pd.DataFrame:
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(f"SELECT * FROM {table}", conn)


def _normalize_timestamp(series: pd.Series) -> pd.Series:
    ts = pd.to_datetime(series, errors="coerce", utc=True).dt.tz_convert("UTC")
    return ts.dt.tz_localize(None)


def _load_full_split(split: str) -> tuple[pd.DataFrame, list[str]]:
    features = _load_table(FEATURES_DB, FEATURE_TABLES[split])
    labels = _load_table(LABELS_DB, LABEL_TABLES[split])
    dst_info = _load_table(DST_DB, DST_TABLES[split])[["timestamp", "dst_dst"]]

    feature_cols = [
        c for c in features.columns
        if c != "timestamp" and np.issubdtype(features[c].dtype, np.number)
    ]
    features["timestamp"] = _normalize_timestamp(features["timestamp"])
    labels["timestamp"] = _normalize_timestamp(labels["timestamp"])
    dst_info["timestamp"] = _normalize_timestamp(dst_info["timestamp"])

    merged = (
        features
        .merge(dst_info, on="timestamp", how="inner")
        .merge(labels, on="timestamp", how="inner")
    )
    if merged.empty:
        raise RuntimeError(f"No merged rows for split '{split}'.")
    return merged, feature_cols


def load_year(year: int) -> tuple[pd.DataFrame, list[str]]:
    frames = []
    feature_cols = None
    for split in ("train", "validation", "test"):
        df, fcols = _load_full_split(split)
        feature_cols = feature_cols or fcols
        frames.append(df)
    if "dst_dst" not in feature_cols:
        feature_cols.append("dst_dst")
    df_all = pd.concat(frames, ignore_index=True)
    df_all = df_all[df_all["timestamp"].dt.year == year].copy()
    if df_all.empty:
        raise RuntimeError(f"No rows found for year {year}.")
    df_all = df_all.dropna(axis=0, how="any")
    return df_all, feature_cols


def plot_year(year: int) -> None:
    regime_lower = REGIME.lower()
    if regime_lower not in {"storm", "calm", "both"}:
        raise ValueError("REGIME must be 'storm', 'calm', or 'both'")

    if not FEATURES_DB.exists():
        raise FileNotFoundError(f"Features DB not found: {FEATURES_DB}")
    if not LABELS_DB.exists():
        raise FileNotFoundError(f"Labels DB not found: {LABELS_DB}")
    if not MODEL_BASE.exists():
        raise FileNotFoundError(f"Model base dir not found: {MODEL_BASE}")

    df_all, feature_cols = load_year(year)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df_all["timestamp"], df_all["h1"], label="DST (actual)", color="black")

    rmses_all: list[float] = []
    regimes = []
    storm_color = "#ff7f7f"  # slightly more intense red
    calm_color = "#1f77b4"   # matplotlib default blue
    if regime_lower in {"storm", "both"}:
        regimes.append(("storm", storm_color, "<="))
    if regime_lower in {"calm", "both"}:
        regimes.append(("calm", calm_color, ">="))

    segments: list[dict] = []
    for regime_name, color, op in regimes:
        rmses = []
        for h in HORIZONS:
            target_col = f"h{h}"
            feat_cols_with_dst = list(feature_cols)
            if "dst_dst" not in feat_cols_with_dst:
                feat_cols_with_dst.append("dst_dst")
            numeric = df_all[feat_cols_with_dst + [target_col]].dropna(axis=0, how="any")
            if target_col not in numeric.columns:
                raise RuntimeError(f"Missing target column '{target_col}' after numeric filtering.")

            if op == "<=":
                mask = numeric["dst_dst"] <= DST_THRESHOLD
            else:
                mask = numeric["dst_dst"] >= DST_THRESHOLD
            numeric_pred = numeric.loc[mask].copy()
            if numeric_pred.empty:
                continue

            segment_ids = numeric_pred.index.to_series().diff().gt(1).cumsum()

            preds = {}
            for q in QUANTILES:
                model_path = MODEL_BASE / f"h{h}_{regime_name}" / f"q{q}" / "model.joblib"
                if not model_path.exists():
                    raise FileNotFoundError(f"Quantile model not found: {model_path}")
                model = joblib.load(model_path)
                preds[q] = model.predict(numeric_pred[feature_cols].to_numpy(dtype=np.float32)).astype(float)

            for _, seg_idx in numeric_pred.groupby(segment_ids).groups.items():
                seg = numeric_pred.loc[seg_idx]
                ts_seg = df_all.loc[seg.index, "timestamp"]
                idx_positions = numeric_pred.index.get_indexer(seg.index)
                q10 = preds[0.1][idx_positions]
                q50 = preds[0.5][idx_positions]
                q90 = preds[0.9][idx_positions]
                ts_shifted = ts_seg + pd.Timedelta(hours=-h)

                y_true = df_all.loc[seg.index, target_col].to_numpy(dtype=float)
                rmse = float(np.sqrt(np.mean((y_true - q50) ** 2)))
                print(f"[RMSE] h{h} segment (REGIME={regime_name}, n={len(y_true)}): {rmse:.4f}")
                rmses.append(rmse)

                segments.append(
                    {
                        "regime": regime_name,
                        "color": color,
                        "h": h,
                        "ts": ts_shifted.reset_index(drop=True),
                        "q10": np.asarray(q10, dtype=float),
                        "q50": np.asarray(q50, dtype=float),
                        "q90": np.asarray(q90, dtype=float),
                        "start": ts_shifted.iloc[0],
                        "end": ts_shifted.iloc[-1],
                    }
                )

        if rmses:
            avg_rmse = float(np.mean(rmses))
            rmses_all.extend(rmses)
            print(f"[RMSE] Average across segments (REGIME={regime_name}): {avg_rmse:.4f}")

    # Create transition segments that linearly interpolate between regimes.
    for h in HORIZONS:
        h_segments = [s for s in segments if s["h"] == h]
        h_segments_sorted = sorted(h_segments, key=lambda s: s["start"])
        for prev, nxt in zip(h_segments_sorted, h_segments_sorted[1:]):
            if prev["regime"] == nxt["regime"]:
                continue
            if nxt["start"] <= prev["end"]:
                continue
            ts_interp = pd.to_datetime(
                np.linspace(prev["end"].value, nxt["start"].value, 3)
            )
            interp = {
                "regime": nxt["regime"],
                "color": nxt["color"],
                "h": h,
                "ts": ts_interp,
                "q10": np.linspace(prev["q10"][-1], nxt["q10"][0], 3),
                "q50": np.linspace(prev["q50"][-1], nxt["q50"][0], 3),
                "q90": np.linspace(prev["q90"][-1], nxt["q90"][0], 3),
                "start": ts_interp[0],
                "end": ts_interp[-1],
                "interp": True,
            }
            segments.append(interp)

    # Plot all segments (original and interpolated), labeling once per regime/horizon.
    label_tracker = {}
    for seg in sorted(segments, key=lambda s: s["start"]):
        key = (seg["regime"], seg["h"])
        band_label = None
        line_label = None
        if not seg.get("interp") and not label_tracker.get(key):
            band_label = f"{seg['regime']} h{seg['h']} q0.1–q0.9"
            line_label = f"{seg['regime']} h{seg['h']} q0.5"
            label_tracker[key] = True

        ax.fill_between(
            seg["ts"],
            seg["q10"],
            seg["q90"],
            color=seg["color"],
            alpha=0.2,
            label=band_label,
        )
        ax.plot(
            seg["ts"],
            seg["q50"],
            color=seg["color"],
            linestyle="--",
            linewidth=1.6,
            label=line_label,
        )

    ax.set_title(f"DST vs forecasts (aligned) - {year}")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("DST")
    ax.legend(ncol=3, fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.autofmt_xdate()
    plt.show()

    if rmses_all:
        avg_rmse = float(np.mean(rmses_all))
        print(f"[RMSE] Overall average across segments: {avg_rmse:.4f}")


if __name__ == "__main__":
    plot_year(YEAR)

from __future__ import annotations

import sqlite3
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DST_DB = PROJECT_ROOT / "inference" / "preprocessed_vector_1m.db"
DST_TABLE = "inference_vector"
FEATURE_DB = PROJECT_ROOT / "inference" / "regression" / "pca_regression_vector_1m.db"
FEATURE_TABLE = "pca_inference_vector"
MODEL_BASE = PROJECT_ROOT / "regression_pipeline" / "xgb_quantile_regime_aware_model"
STORM_COLOR = "#ff7f7f"
CALM_COLOR = "#1f77b4"

HISTORY_HOURS = 24 * 7  # hours of history to show
FUTURE_HOURS = 6    # hours ahead to show
QUANTILES = [0.1, 0.5, 0.9]
DST_THRESHOLD = -20.0

TIMESTAMP_COLS = ["timestamp", "time_tag", "date"]
DST_TARGET_CANDIDATES = ["h1", "dst", "Dst", "dst_value"]


# Helper to load a SQLite table into a pandas DataFrame
def _load_table(db_path: Path, table: str) -> pd.DataFrame:
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(f"SELECT * FROM {table}", conn)


# Standardize timestamps across different data sources to UTC
def _normalize_ts(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, errors="coerce")


# Search for known timestamp column names in a DataFrame
def _detect_ts(df: pd.DataFrame) -> str:
    for c in TIMESTAMP_COLS:
        if c in df.columns:
            return c
    return None


# Search for known Dst target column names or fallback to the first numeric column
def _detect_dst(df: pd.DataFrame) -> str:
    for c in DST_TARGET_CANDIDATES:
        if c in df.columns:
            return c
    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    return numeric[0] if numeric else None


# Main logic to generate the Dst prediction plot with regime-aware quantile forecasts
def main() -> None:
    # 1. Verification of recursive pipeline dependencies
    if not DST_DB.exists():
        raise FileNotFoundError(f"DST DB not found: {DST_DB}")
    if not FEATURE_DB.exists():
        raise FileNotFoundError(f"Feature DB not found: {FEATURE_DB}")
    if not MODEL_BASE.exists():
        raise FileNotFoundError(f"Model base dir not found: {MODEL_BASE}")

    # 2. Data Loading and Normalization
    dst_df = _load_table(DST_DB, DST_TABLE)
    ts_col_dst = _detect_ts(dst_df)
    dst_col = _detect_dst(dst_df)
    if not ts_col_dst or not dst_col:
        raise RuntimeError(
            f"DST table must contain a timestamp column and a numeric target. "
            f"Found columns: {list(dst_df.columns)}"
        )
    dst_df[ts_col_dst] = _normalize_ts(dst_df[ts_col_dst])
    dst_df = dst_df.dropna(subset=[ts_col_dst, dst_col]).sort_values(ts_col_dst)

    # 3. Pipeline Anchor Points
    # Identify the latest record as the T0 for future forecasting
    anchor_ts = dst_df[ts_col_dst].max()
    if pd.isna(anchor_ts):
        raise RuntimeError("No valid timestamps in DST data.")
    anchor_dst = dst_df.loc[dst_df[ts_col_dst] == anchor_ts, dst_col].iloc[-1]
    
    # Determine if we are currently in a 'storm' or 'calm' state based on the Dst threshold
    regime = "storm" if anchor_dst <= DST_THRESHOLD else "calm"

    # Slice historical data for the plot view (e.g., last 7 days)
    hist_start = anchor_ts - pd.Timedelta(hours=HISTORY_HOURS)
    dst_hist = dst_df[dst_df[ts_col_dst] >= hist_start]

    # 4. Feature and Model Preparation
    feat_df = _load_table(FEATURE_DB, FEATURE_TABLE)
    ts_col_feat = _detect_ts(feat_df)
    if not ts_col_feat:
        raise RuntimeError(f"Feature table '{FEATURE_TABLE}' lacks a timestamp column.")
    feat_df[ts_col_feat] = _normalize_ts(feat_df[ts_col_feat])
    feat_df = feat_df.dropna(subset=[ts_col_feat]).sort_values(ts_col_feat)
    
    # Merge existing Dst as a feature for the regime-aware regression
    feat_df = feat_df.merge(
        dst_df[[ts_col_dst, dst_col]].rename(columns={ts_col_dst: ts_col_feat, dst_col: "dst_dst"}),
        on=ts_col_feat,
        how="left",
    )
    feature_cols = [
        c for c in feat_df.columns
        if c != ts_col_feat and pd.api.types.is_numeric_dtype(feat_df[c])
    ]
    if "dst_dst" not in feature_cols:
        raise RuntimeError("Expected 'dst_dst' to be present after merge for regime-aware model.")
    
    # Extract the precise feature vector for the prediction anchor
    anchor_row = feat_df[feat_df[ts_col_feat] == anchor_ts]
    if anchor_row.empty:
        raise RuntimeError(f"No feature row found at anchor timestamp {anchor_ts}.")
    x_anchor = anchor_row.iloc[0][feature_cols].to_numpy(dtype=np.float32).reshape(1, -1)

    # Lazy-loader for regime-specific XGBoost quantile models
    model_cache: dict[tuple[str, int, float], object] = {}

    def _predict(regime: str, h: int, q: float, x: np.ndarray) -> float | None:
        key = (regime, h, q)
        if key not in model_cache:
            model_path = MODEL_BASE / f"h{h}_{regime}" / f"q{q}" / "model.joblib"
            if not model_path.exists():
                return None
            # Quantile models are organized by horizon (h), regime (calm/storm), and quantile (q)
            model_cache[key] = joblib.load(model_path)
        model = model_cache[key]
        return float(model.predict(x)[0])

    # 5. Quantile Back-forecasting (History Verification)
    # Generate 1h-ahead predictions for all past timestamps to visualize model performance in the past
    history_preds = {
        "storm": {"t": [], "q10": [], "q50": [], "q90": [], "color": STORM_COLOR},
        "calm": {"t": [], "q10": [], "q50": [], "q90": [], "color": CALM_COLOR},
    }
    hist_feat = feat_df[feat_df[ts_col_feat] >= hist_start].dropna(subset=feature_cols + ["dst_dst"])
    for _, row in hist_feat.iterrows():
        regime_row = "storm" if row["dst_dst"] <= DST_THRESHOLD else "calm"
        x_row = row[feature_cols].to_numpy(dtype=np.float32).reshape(1, -1)
        q10 = _predict(regime_row, 1, 0.1, x_row)
        q50 = _predict(regime_row, 1, 0.5, x_row)
        q90 = _predict(regime_row, 1, 0.9, x_row)
        if q10 is None or q50 is None or q90 is None:
            continue
        # Issued forecasts align with their actual occurrence time
        t_plot = row[ts_col_feat]
        history_preds[regime_row]["t"].append(t_plot)
        history_preds[regime_row]["q10"].append(q10)
        history_preds[regime_row]["q50"].append(q50)
        history_preds[regime_row]["q90"].append(q90)

    # 6. Future Forecast Generation
    forecast_times = []
    q10_list = []
    q50_list = []
    q90_list = []
    color = STORM_COLOR if regime == "storm" else CALM_COLOR

    # Predict future horizons (h=1 to FUTURE_HOURS)
    for h in range(1, FUTURE_HOURS + 1):
        preds = {}
        for q in QUANTILES:
            pred_val = _predict(regime, h, q, x_anchor)
            if pred_val is None:
                preds = {}
                break
            preds[q] = pred_val
        if not preds:
            continue
        forecast_times.append(anchor_ts + pd.Timedelta(hours=h))
        q10_list.append(preds[0.1])
        q50_list.append(preds[0.5])
        q90_list.append(preds[0.9])

    # 7. Visualization logic
    fig, ax = plt.subplots(figsize=(10, 5))
    actual_color = "black"
    # Plot the observed Dst line
    ax.plot(dst_hist[ts_col_dst], dst_hist[dst_col], label="DST (actual)", color=actual_color)

    # Plot future forecast bands and 50th percentile (median) prediction
    if forecast_times:
        ax.fill_between(
            forecast_times,
            q10_list,
            q90_list,
            color=color,
            alpha=0.2,
            linewidth=0,
            edgecolor="none",
            step=None,
            label=None,
        )
        ax.plot(
            forecast_times,
            q50_list,
            color=color,
            linestyle="--",
            label=f"DST forecast $q_{{0.5}}$ ({regime})",
        )
        # Bridge the gap between last actual and first forecast with a black interpolation to show the trend
        ax.plot(
            [anchor_ts, forecast_times[0]],
            [anchor_dst, q50_list[0]],
            color="black",
            linewidth=1.2,
            linestyle="-",
            label=None,
        )

    # 8. Historical Segment Interpolation Logic
    # Dst shifts between regimes cause gaps in history; find continuous segments and bridge them visually
    hist_segments = []
    gap = pd.Timedelta(hours=1.5)
    for reg_name, data in history_preds.items():
        if not data["t"]:
            continue
        sort_idx = np.argsort(data["t"])
        t_sorted = np.array(data["t"])[sort_idx]
        q10_sorted = np.array(data["q10"])[sort_idx]
        q90_sorted = np.array(data["q90"])[sort_idx]

        start = 0
        for i in range(1, len(t_sorted)):
            # If timestamps are separated by more than the gap threshold, break into a new segment
            if t_sorted[i] - t_sorted[i - 1] > gap:
                seg = slice(start, i)
                hist_segments.append(
                    {
                        "regime": reg_name,
                        "color": data["color"],
                        "ts": t_sorted[seg],
                        "q10": q10_sorted[seg],
                        "q90": q90_sorted[seg],
                        "start": t_sorted[seg][0],
                        "end": t_sorted[seg][-1],
                        "interp": False,
                    }
                )
                start = i
        seg = slice(start, len(t_sorted))
        hist_segments.append(
            {
                "regime": reg_name,
                "color": data["color"],
                "ts": t_sorted[seg],
                "q10": q10_sorted[seg],
                "q90": q90_sorted[seg],
                "start": t_sorted[seg][0],
                "end": t_sorted[seg][-1],
                "interp": False,
            }
        )

    # Sort all segments by time and interpolate gaps between different regimes
    def _as_ts(val):
        return pd.to_datetime(val).tz_localize(None) if hasattr(val, "tzinfo") else pd.to_datetime(val)

    hist_segments_sorted = sorted(hist_segments, key=lambda s: _as_ts(s["start"]))
    for prev, nxt in zip(hist_segments_sorted, hist_segments_sorted[1:]):
        if prev["regime"] == nxt["regime"]:
            continue
        if _as_ts(nxt["start"]) <= _as_ts(prev["end"]):
            continue
        # Linear interpolation for the uncertainty bands across regime transitions
        ts_interp = pd.to_datetime(np.linspace(_as_ts(prev["end"]).value, _as_ts(nxt["start"]).value, 3))
        hist_segments.append(
            {
                "regime": nxt["regime"],
                "color": nxt["color"],
                "ts": ts_interp,
                "q10": np.linspace(prev["q10"][-1], nxt["q10"][0], 3),
                "q90": np.linspace(prev["q90"][-1], nxt["q90"][0], 3),
                "start": ts_interp[0],
                "end": ts_interp[-1],
                "interp": True,
            }
        )
    # Final interpolation from the last historical segment to the start of the future forecast
    if hist_segments and forecast_times:
        last_seg = max(hist_segments, key=lambda s: _as_ts(s["end"]))
        first_future_ts = pd.to_datetime(forecast_times[0]).tz_localize(None)
        if _as_ts(last_seg["end"]) < first_future_ts:
            ts_bridge = pd.to_datetime(
                np.linspace(_as_ts(last_seg["end"]).value, first_future_ts.value, 3)
            )
            hist_segments.append(
                {
                    "regime": regime,
                    "color": color,
                    "ts": ts_bridge,
                    "q10": np.linspace(last_seg["q10"][-1], q10_list[0], 3),
                    "q90": np.linspace(last_seg["q90"][-1], q90_list[0], 3),
                    "start": ts_bridge[0],
                    "end": ts_bridge[-1],
                    "interp": True,
                }
            )

    # 9. Plotting all segments with cumulative labeling
    label_used = {}
    for seg in sorted(hist_segments, key=lambda s: _as_ts(s["start"])):
        key = seg["regime"]
        lbl = None if label_used.get(key) else f"Issued 1h $q_{{0.1}}$–$q_{{0.9}}$ ({key})"
        ax.fill_between(
            seg["ts"],
            seg["q10"],
            seg["q90"],
            color=seg["color"],
            alpha=0.2,
            linewidth=0,
            edgecolor="none",
            step=None,
            label=lbl,
        )
        label_used[key] = True

    # 10. Finishing touches and persistence
    ax.set_title("DST with forecast")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("DST")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.autofmt_xdate()
    latest_ts = dst_df[ts_col_dst].max()
    out_path = PROJECT_ROOT / "inference" / "regression" / "predicted_dst.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved DST forecast plot to {out_path} (latest timestamp: {latest_ts})")


if __name__ == "__main__":
    main()

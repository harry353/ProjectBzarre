from __future__ import annotations

import sqlite3
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DST_DB = PROJECT_ROOT / "inference" / "preprocessed_vector_1m.db"
DST_TABLE = "inference_vector"
FEATURE_DB = PROJECT_ROOT / "inference" / "regression" / "pca_regression_vector_1m.db"
FEATURE_TABLE = "pca_inference_vector"
MODEL_BASE = PROJECT_ROOT / "regression_pipeline" / "xgb_quantile_regime_aware_model"

PROB_DB = PROJECT_ROOT / "inference" / "classification" / "classification_predictions.db"
PROB_TABLE = "predictions"
PROB_TS_COL = "timestamp"

STORM_COLOR = "#ff7f7f"
CALM_COLOR = "#1f77b4"

HISTORY_HOURS = 24 * 30
FUTURE_HOURS = 6
QUANTILES = [0.1, 0.5, 0.9]
DST_THRESHOLD = -20.0
ZOOMED_IN_DAYS = 3

TIMESTAMP_COLS = ["timestamp", "time_tag", "date"]
DST_TARGET_CANDIDATES = ["h1", "dst", "Dst", "dst_value", "dst_dst"]


def _load_table(db_path: Path, table: str) -> pd.DataFrame:
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(f"SELECT * FROM {table}", conn)


def _normalize_ts(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, errors="coerce")


def _detect_ts(df: pd.DataFrame) -> str | None:
    for c in TIMESTAMP_COLS:
        if c in df.columns:
            return c
    return None


def _detect_dst(df: pd.DataFrame) -> str | None:
    for c in DST_TARGET_CANDIDATES:
        if c in df.columns:
            return c
    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    return numeric[0] if numeric else None


def main() -> None:
    if not DST_DB.exists():
        raise FileNotFoundError(f"DST DB not found: {DST_DB}")
    if not FEATURE_DB.exists():
        raise FileNotFoundError(f"Feature DB not found: {FEATURE_DB}")
    if not MODEL_BASE.exists():
        raise FileNotFoundError(f"Model base dir not found: {MODEL_BASE}")
    if not PROB_DB.exists():
        raise FileNotFoundError(f"Classification predictions DB not found: {PROB_DB}")

    # Load DST
    dst_df = _load_table(DST_DB, DST_TABLE)
    ts_col_dst = _detect_ts(dst_df)
    dst_col = _detect_dst(dst_df)
    if not ts_col_dst or not dst_col:
        raise RuntimeError(
            f"DST table must contain a timestamp column and a numeric target. "
            f"Found columns: {list(dst_df.columns)}"
        )
    dst_df[ts_col_dst] = _normalize_ts(dst_df[ts_col_dst]).dt.tz_localize(None)
    dst_df = dst_df.dropna(subset=[ts_col_dst, dst_col]).sort_values(ts_col_dst)

    anchor_ts = dst_df[ts_col_dst].max()
    if pd.isna(anchor_ts):
        raise RuntimeError("No valid timestamps in DST data.")
    anchor_ts_naive = anchor_ts.tz_convert(None) if getattr(anchor_ts, "tzinfo", None) else anchor_ts
    anchor_ts = anchor_ts_naive
    anchor_dst = dst_df.loc[dst_df[ts_col_dst] == anchor_ts, dst_col].iloc[-1]
    regime = "storm" if anchor_dst <= DST_THRESHOLD else "calm"

    hist_start = anchor_ts_naive - pd.Timedelta(hours=HISTORY_HOURS)
    dst_hist = dst_df[dst_df[ts_col_dst] >= hist_start]

    # Load features and merge dst_dst
    feat_df = _load_table(FEATURE_DB, FEATURE_TABLE)
    ts_col_feat = _detect_ts(feat_df)
    if not ts_col_feat:
        raise RuntimeError(f"Feature table '{FEATURE_TABLE}' lacks a timestamp column.")
    feat_df[ts_col_feat] = _normalize_ts(feat_df[ts_col_feat]).dt.tz_localize(None)
    feat_df = feat_df.dropna(subset=[ts_col_feat]).sort_values(ts_col_feat)
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
    anchor_row = feat_df[feat_df[ts_col_feat] == anchor_ts]
    if anchor_row.empty:
        raise RuntimeError(f"No feature row found at anchor timestamp {anchor_ts}.")
    x_anchor = anchor_row.iloc[0][feature_cols].to_numpy(dtype=np.float32).reshape(1, -1)

    model_cache: dict[tuple[str, int, float], object] = {}

    def _predict(reg: str, h: int, q: float, x: np.ndarray) -> float | None:
        key = (reg, h, q)
        if key not in model_cache:
            model_path = MODEL_BASE / f"h{h}_{reg}" / f"q{q}" / "model.joblib"
            if not model_path.exists():
                return None
            model_cache[key] = joblib.load(model_path)
        model = model_cache[key]
        return float(model.predict(x)[0])

    # Past issued forecasts (shifted back to issue time)
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
        t_plot = row[ts_col_feat]
        history_preds[regime_row]["t"].append(t_plot)
        history_preds[regime_row]["q10"].append(q10)
        history_preds[regime_row]["q50"].append(q50)
        history_preds[regime_row]["q90"].append(q90)

    # Future forecasts from anchor
    forecast_times = []
    q10_list = []
    q50_list = []
    q90_list = []
    color = STORM_COLOR if regime == "storm" else CALM_COLOR

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

    # Build contiguous segments of past issued forecasts and interpolate across regime changes
    def _as_ts(val):
        ts = pd.to_datetime(val)
        return ts.tz_localize(None) if getattr(ts, "tzinfo", None) else ts

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

    hist_segments_sorted = sorted(hist_segments, key=lambda s: _as_ts(s["start"]))
    for prev, nxt in zip(hist_segments_sorted, hist_segments_sorted[1:]):
        if prev["regime"] == nxt["regime"]:
            continue
        if _as_ts(nxt["start"]) <= _as_ts(prev["end"]):
            continue
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
    # Interpolate from last past segment to first future forecast point
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

    fig, ((ax_dst, ax_dst_zoom), (ax_prob, ax_prob_zoom)) = plt.subplots(
        2,
        2,
        figsize=(14, 8),
        sharex="col",
        sharey="row",
        gridspec_kw={"width_ratios": [2, 1], "wspace": 0.04},
    )

    # DST history and forecasts (matching plot_predicted_dst)
    ax_dst.plot(
        dst_hist[ts_col_dst],
        dst_hist[dst_col],
        label="DST (actual)",
        color="black",
        linewidth=1.2,
    )

    if forecast_times:
        ax_dst.fill_between(
            forecast_times,
            q10_list,
            q90_list,
            color=color,
            alpha=0.3,
            linewidth=0,
            edgecolor="none",
            step=None,
            label=None,
        )
        ax_dst.plot(
            forecast_times,
            q50_list,
            color=color,
            linestyle="--",
            label=f"DST forecast $q_{{0.5}}$ ({regime})",
        )
        ax_dst.plot(
            [anchor_ts, forecast_times[0]],
            [anchor_dst, q50_list[0]],
            color="black",
            linewidth=1.2,
            linestyle="-",
            label=None,
        )

    label_used = {}
    for seg in sorted(hist_segments, key=lambda s: _as_ts(s["start"])):
        key = seg["regime"]
        lbl = None if label_used.get(key) else f"Issued $q_{{0.1}}$–$q_{{0.9}}$ ({key})"
        ax_dst.fill_between(
            seg["ts"],
            seg["q10"],
            seg["q90"],
            color=seg["color"],
            alpha=0.3,
            linewidth=0,
            edgecolor="none",
            step=None,
            label=lbl,
        )
        label_used[key] = True

    ax_dst.set_title("DST with forecast")
    ax_dst.set_ylabel("Dst (nT)")
    ax_dst.grid(True, linestyle="--", alpha=0.4)
    ax_dst.legend()

    # Probability plot (full range)
    prob_df = _load_table(PROB_DB, PROB_TABLE)
    if PROB_TS_COL not in prob_df:
        raise RuntimeError(f"Prediction table missing timestamp column '{PROB_TS_COL}'.")
    prob_df[PROB_TS_COL] = _normalize_ts(prob_df[PROB_TS_COL]).dt.tz_localize(None)
    prob_df = prob_df.dropna(subset=[PROB_TS_COL, "p_cumulative"]).sort_values(PROB_TS_COL)
    prob_df = prob_df[prob_df[PROB_TS_COL] >= hist_start]
    prob_df = prob_df.rename(columns={PROB_TS_COL: "timestamp"})

    ax_prob.plot(
        prob_df["timestamp"],
        prob_df["p_cumulative"],
        color="black",
        marker="o",
        markersize=2,
        linewidth=1.2,
        linestyle="-",
        label="Storm probability",
    )
    ax_prob.set_title("Storm probability")
    ax_prob.set_xlabel("Timestamp")
    ax_prob.set_ylabel("Probability")
    ax_prob.grid(True, linestyle="--", alpha=0.4)
    ax_prob.legend()

    # Zoomed views (last 7 days)
    zoom_start = anchor_ts_naive - pd.Timedelta(days=ZOOMED_IN_DAYS)
    dst_zoom = dst_df[dst_df[ts_col_dst] >= zoom_start]
    ax_dst_zoom.plot(dst_zoom[ts_col_dst], dst_zoom[dst_col], label="DST (actual)", color="black", linewidth=1.2)

    forecast_mask = [ _as_ts(t) >= zoom_start for t in forecast_times]
    if forecast_times and any(forecast_mask):
        ft_zoom = [t for t, m in zip(forecast_times, forecast_mask) if m]
        q10_zoom = [v for v, m in zip(q10_list, forecast_mask) if m]
        q50_zoom = [v for v, m in zip(q50_list, forecast_mask) if m]
        q90_zoom = [v for v, m in zip(q90_list, forecast_mask) if m]
        ax_dst_zoom.fill_between(
            ft_zoom,
            q10_zoom,
            q90_zoom,
            color=color,
            alpha=0.3,
            linewidth=0,
            edgecolor="none",
            step=None,
            label=None,
        )
        ax_dst_zoom.plot(
            ft_zoom,
            q50_zoom,
            color=color,
            linestyle="--",
            label=f"DST forecast $q_{{0.5}}$ ({regime})",
        )
        if ft_zoom:
            ax_dst_zoom.plot(
                [anchor_ts, ft_zoom[0]],
                [anchor_dst, q50_zoom[0]],
                color="black",
                linewidth=1.2,
                linestyle="-",
                label=None,
            )

    label_used_zoom = {}
    for seg in sorted(hist_segments, key=lambda s: _as_ts(s["start"])):
        if _as_ts(seg["end"]) < zoom_start:
            continue
        ts_seg = [t for t in seg["ts"] if _as_ts(t) >= zoom_start]
        if not ts_seg:
            continue
        idxs = [i for i, t in enumerate(seg["ts"]) if _as_ts(t) >= zoom_start]
        q10_seg = np.array(seg["q10"])[idxs]
        q90_seg = np.array(seg["q90"])[idxs]
        key = seg["regime"]
        lbl = None if label_used_zoom.get(key) else f"Issued $q_{{0.1}}$–$q_{{0.9}}$ ({key})"
        ax_dst_zoom.fill_between(
            ts_seg,
            q10_seg,
            q90_seg,
            color=seg["color"],
            alpha=0.3,
            linewidth=0,
            edgecolor="none",
            step=None,
            label=lbl,
        )
        label_used_zoom[key] = True

    ax_dst_zoom.set_title(f"DST (last {ZOOMED_IN_DAYS} days)")
    ax_dst_zoom.grid(True, linestyle="--", alpha=0.4)
    prob_zoom = prob_df[prob_df["timestamp"] >= zoom_start]
    ax_prob_zoom.plot(
        prob_zoom["timestamp"],
        prob_zoom["p_cumulative"],
        color="black",
        marker="o",
        markersize=2,
        linewidth=1.2,
        linestyle="-",
        label=None,
    )
    ax_prob_zoom.set_title(f"Storm probability (last {ZOOMED_IN_DAYS} days)")
    ax_prob_zoom.set_xlabel("Timestamp")
    ax_prob_zoom.grid(True, linestyle="--", alpha=0.4)

    # Align x limits across both axes
    x_min_candidates = []
    x_max_candidates = []
    if not dst_df.empty:
        x_min_candidates.append(_as_ts(dst_df[ts_col_dst].min()))
        x_max_candidates.append(_as_ts(dst_df[ts_col_dst].max()))
    if not prob_df.empty:
        x_min_candidates.append(_as_ts(prob_df["timestamp"].min()))
        x_max_candidates.append(_as_ts(prob_df["timestamp"].max()))
    if forecast_times:
        x_min_candidates.append(_as_ts(min(forecast_times)))
        x_max_candidates.append(_as_ts(max(forecast_times)))
    if hist_segments:
        x_min_candidates.append(min(_as_ts(seg["start"]) for seg in hist_segments))
        x_max_candidates.append(max(_as_ts(seg["end"]) for seg in hist_segments))
    if x_min_candidates and x_max_candidates:
        x_min = min(x_min_candidates)
        x_max = max(x_max_candidates) + pd.Timedelta(hours=8)
        ax_dst.set_xlim(x_min, x_max)
        ax_prob.set_xlim(x_min, x_max)
        ax_dst_zoom.set_xlim(zoom_start, anchor_ts_naive + pd.Timedelta(hours=8))
        ax_prob_zoom.set_xlim(zoom_start, anchor_ts_naive + pd.Timedelta(hours=8))

    fig.autofmt_xdate()
    latest_ts_str = "unavailable"
    if anchor_ts is not None:
        ts_for_title = anchor_ts.tz_convert("UTC") if anchor_ts.tzinfo else anchor_ts
        latest_ts_str = ts_for_title.strftime("%Y-%m-%d %H:%M:%S %Z").strip()
        fig.suptitle(f"Latest observation: {latest_ts_str} UTC")
    else:
        fig.suptitle("Latest observation: unavailable")

    out_path = PROJECT_ROOT / "inference" / "combined_predicted_dst_and_prob.png"
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"[OK] Saved combined plot to {out_path} (latest DST timestamp: {latest_ts_str})")


if __name__ == "__main__":
    main()

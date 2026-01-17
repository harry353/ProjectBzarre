from __future__ import annotations

import sqlite3
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # ensure no GUI needed
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRED_DB = PROJECT_ROOT / "inference" / "horizon_predictions.db"
DST_DB = PROJECT_ROOT / "inference" / "inference_vector.db"
OUTPUT_PATH = PROJECT_ROOT / "inference" / "predicted_probabilities.png"
TIMESTAMP_COLS = ["timestamp", "time_tag", "date"]


def _load_predictions() -> pd.DataFrame:
    if not PRED_DB.exists():
        raise FileNotFoundError(f"Prediction DB not found: {PRED_DB}")

    with sqlite3.connect(PRED_DB) as conn:
        df = pd.read_sql_query("SELECT * FROM predictions", conn)
    if df.empty:
        raise RuntimeError("No predictions found in table 'predictions'.")

    ts_col = next((c for c in TIMESTAMP_COLS if c in df.columns), None)
    if ts_col:
        df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
        df = df.dropna(subset=[ts_col]).sort_values(ts_col)
        df = df.rename(columns={ts_col: "timestamp"})
    else:
        df["timestamp"] = range(len(df))

    if "p_cumulative" not in df.columns:
        raise RuntimeError("Column 'p_cumulative' is missing from predictions.")

    return df[["timestamp", "p_cumulative"]]


def _load_dst() -> pd.DataFrame:
    if not DST_DB.exists():
        raise FileNotFoundError(f"DST DB not found: {DST_DB}")
    with sqlite3.connect(DST_DB) as conn:
        df = pd.read_sql_query("SELECT * FROM inference_vector", conn)
    if df.empty:
        raise RuntimeError("No DST data found in table 'inference_vector'.")
    ts_col = next((c for c in TIMESTAMP_COLS if c in df.columns), None)
    if ts_col:
        df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
        df = df.dropna(subset=[ts_col]).sort_values(ts_col)
        df = df.rename(columns={ts_col: "timestamp"})
    else:
        df["timestamp"] = range(len(df))
    # Prefer dst column name variants
    for cand in ["dst", "dst_index", "dst_dst", "value"]:
        if cand in df.columns:
            df = df.rename(columns={cand: "dst_value"})
            break
    if "dst_value" not in df.columns:
        raise RuntimeError("DST value column not found.")
    return df[["timestamp", "dst_value"]]


def _load_imf() -> pd.DataFrame:
    if not DST_DB.exists():
        raise FileNotFoundError(f"Inference DB not found: {DST_DB}")
    with sqlite3.connect(DST_DB) as conn:
        df = pd.read_sql_query("SELECT * FROM inference_vector", conn)
    if df.empty:
        raise RuntimeError("No IMF data found in table 'inference_vector'.")
    ts_col = next((c for c in TIMESTAMP_COLS if c in df.columns), None)
    if ts_col:
        df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
        df = df.dropna(subset=[ts_col]).sort_values(ts_col)
        df = df.rename(columns={ts_col: "timestamp"})
    else:
        df["timestamp"] = range(len(df))

    bz_col = next((c for c in ["bz", "bz_gse", "imf_solar_wind_bz_gse"] if c in df.columns), None)
    bt_col = next((c for c in ["bt", "imf_solar_wind_bt"] if c in df.columns), None)
    if not bz_col or not bt_col:
        raise RuntimeError("IMF Bz or |B| column not found.")
    df = df.rename(columns={bz_col: "bz", bt_col: "bt"})
    return df[["timestamp", "bz", "bt"]]


def _plot_dst(ax, dst: pd.DataFrame) -> None:
    ax.plot(dst["timestamp"], dst["dst_value"], color="C0", label="DST")
    ax.set_ylabel("Dst (nT)")
    ax.axhline(0, color="0.2", linestyle="-", linewidth=1.0)
    ax.axhline(-50, color="0.2", linestyle="--", linewidth=1.0)
    ax.set_title("DST")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()


def _plot_imf(ax, imf: pd.DataFrame) -> None:
    ax.plot(imf["timestamp"], imf["bz"], color="C2", label="Bz (nT)")
    ax.plot(imf["timestamp"], imf["bt"], color="C3", label="|B| (nT)")
    ax.set_ylabel("B (nT)")
    ax.set_title("IMF Bz and |B|")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()


def _plot_probs(ax, preds: pd.DataFrame) -> None:
    ax.plot(
        preds["timestamp"],
        preds["p_cumulative"],
        marker="o",
        linestyle="-",
        markersize=3,
        color="C0",
        label="Calibrated cumulative prob",
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Probability")
    ax.set_title(r"Cumulative probability for storm in $\leq$ 6h")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    ax.set_ylim(top=1)


def _apply_time_limits(fig, axes, series_list: list[pd.Series]) -> None:
    locator = mdates.DayLocator(interval=1)
    formatter = mdates.DateFormatter("%b-%d")
    axes[-1].xaxis.set_major_locator(locator)
    axes[-1].xaxis.set_major_formatter(formatter)
    fig.autofmt_xdate(rotation=45)

    xmin = None
    xmax = None
    for series in series_list:
        try:
            s = pd.to_datetime(series)
        except Exception:
            continue
        s = s.dropna()
        if s.empty:
            continue
        smin, smax = s.min(), s.max()
        xmin = smin if xmin is None else min(xmin, smin)
        xmax = smax if xmax is None else max(xmax, smax)
    if xmin is not None and xmax is not None:
        xmax_padded = xmax + pd.Timedelta(days=1)
        for ax in axes:
            ax.set_xlim(xmin, xmax_padded)


def plot() -> Path:
    preds = _load_predictions()
    dst = _load_dst()
    imf = _load_imf()

    plotters = []
    # Comment out any of these lines to omit that subplot entirely
    plotters.append(("dst", dst, _plot_dst))
    # plotters.append(("imf", imf, _plot_imf))
    plotters.append(("probs", preds, _plot_probs))

    fig, axes = plt.subplots(len(plotters), 1, figsize=(10, 8), sharex=True)
    if len(plotters) == 1:
        axes = [axes]

    series_for_limits = []
    for ax, (_, data, func) in zip(axes, plotters):
        func(ax, data)
        series_for_limits.append(data["timestamp"])

    _apply_time_limits(fig, axes, series_for_limits)

    fig.tight_layout()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=200)
    plt.close(fig)
    return OUTPUT_PATH


def main() -> None:
    out = plot()
    print(f"[OK] Plot saved to {out}")


if __name__ == "__main__":
    main()

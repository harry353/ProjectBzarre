from __future__ import annotations

import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = PROJECT_ROOT / "ml_pipeline" / "horizon_models"

DST_DB = PROJECT_ROOT / "preprocessing_pipeline" / "dst" / "1_averaging" / "dst_aver.db"

CALIB_TABLE = "calibrated_probs"
HOURS_AHEAD_PREDICTION = 6
TARGET_HORIZONS_H = range(1, HOURS_AHEAD_PREDICTION + 1)

YEAR_TO_PLOT = 2024
SPLIT = "test"
OUTPUT_PATH: Path | None = None


def _ensure_utc(series: pd.Series) -> pd.Series:
    return (
        series.dt.tz_localize("UTC")
        if series.dt.tz is None
        else series.dt.tz_convert("UTC")
    )


def _load_dst() -> pd.Series:
    with sqlite3.connect(DST_DB) as conn:
        df = pd.read_sql_query(
            "SELECT time_tag AS timestamp, dst FROM hourly_data",
            conn,
            parse_dates=["timestamp"],
        )
    df["timestamp"] = _ensure_utc(df["timestamp"])
    return df.set_index("timestamp")["dst"].sort_index()


def _load_calibrated_probs() -> pd.DataFrame:
    frames = []
    for h in TARGET_HORIZONS_H:
        db = MODEL_ROOT / f"h{h}" / "calibrated_probabilities.db"
        if not db.exists():
            continue
        with sqlite3.connect(db) as conn:
            df = pd.read_sql_query(
                f"""
                SELECT timestamp, prob_calibrated, split
                FROM {CALIB_TABLE}
                """,
                conn,
                parse_dates=["timestamp"],
            )
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df[df["split"] == SPLIT].copy()
        df = df.rename(columns={"prob_calibrated": f"p_h{h}"})
        frames.append(df[["timestamp", f"p_h{h}"]])

    if not frames:
        raise RuntimeError("No calibrated probability tables found.")

    out = frames[0]
    for f in frames[1:]:
        out = out.merge(f, on="timestamp", how="inner")

    return out.sort_values("timestamp").reset_index(drop=True)


def _compute_cumulative_probability(probs: pd.DataFrame) -> pd.Series:
    hazard_cols = [f"p_h{h}" for h in TARGET_HORIZONS_H if f"p_h{h}" in probs.columns]
    surv = 1.0
    for col in hazard_cols:
        surv *= 1.0 - probs[col]
    return 1.0 - surv


def plot_cumulative_probability(year: int, output: Path | None = None) -> None:
    dst = _load_dst()
    probs = _load_calibrated_probs()

    start = pd.Timestamp(year=year, month=1, day=1, tz="UTC")
    end = start + pd.DateOffset(years=1)

    dst = dst.loc[start:end]
    probs = probs.loc[
        (probs["timestamp"] >= start) & (probs["timestamp"] < end)
    ]

    probs["p_cumulative"] = _compute_cumulative_probability(probs)

    fig, (ax_dst, ax_prob) = plt.subplots(
        2, 1, figsize=(14, 6), sharex=True, height_ratios=[2, 1]
    )

    ax_dst.plot(dst.index, dst, color="black", linewidth=1.2)
    ax_dst.axhline(0, color="gray", alpha=0.5)
    ax_dst.axhline(-50, color="gray", linestyle=":", alpha=0.4)

    ax_prob.plot(
        probs["timestamp"],
        probs["p_cumulative"],
        color="tab:red",
        linewidth=1.5,
    )

    ax_dst.set_ylabel("Dst (nT)")
    ax_prob.set_ylabel(f"P(storm within next {HOURS_AHEAD_PREDICTION}h)")
    ax_prob.set_xlabel("Time")

    ax_dst.set_title(
        f"DST and cumulative storm probability ({SPLIT}) — {year}"
    )

    ax_dst.grid(alpha=0.3)
    ax_prob.grid(alpha=0.3)

    ax_prob.legend(
        handles=[Line2D([0], [0], color="tab:red", label=f"≤ {HOURS_AHEAD_PREDICTION}h cumulative")],
        loc="upper right",
    )

    fig.tight_layout()

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=200)
    else:
        plt.show()

    plt.close(fig)


def main() -> None:
    plot_cumulative_probability(YEAR_TO_PLOT, OUTPUT_PATH)


if __name__ == "__main__":
    main()

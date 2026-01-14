from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

# ---------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------

START = "2024-01-01"
END   = "2024-12-31"   # change freely

# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def ensure_utc(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
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
    df.index = ensure_utc(df.index)
    return df


def load_dst() -> pd.DataFrame:
    with sqlite3.connect(DST_DB) as conn:
        df = pd.read_sql(
            "SELECT time_tag, dst FROM hourly_data",
            conn,
            parse_dates=["time_tag"],
        )
    df = df.set_index("time_tag").sort_index()
    df.index = ensure_utc(df.index)
    return df

# ---------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------

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

    X["bz_neg_6h_frac"] = (df["bz_gse"] < -5).rolling(6).mean()
    X["bz_neg_12h_frac"] = (df["bz_gse"] < -5).rolling(12).mean()

    X["clock_angle_south_frac_6h"] = (
        (np.abs(X["clock_angle"] - 180) < 30).rolling(6).mean()
    )

    X["ey_integral_6h"] = ey.rolling(6).sum()

    X["dp_jump"] = df["dynamic_pressure"].diff()
    X["bt_jump"] = df["bt"].diff()

    X["shock_like"] = (
        (df["dynamic_pressure"].diff() > 2) &
        (df["bt"].diff() > 3)
    ).astype(int)

    return X

# ---------------------------------------------------------------------
# Main plotting routine
# ---------------------------------------------------------------------

def main() -> None:
    imf = load_imf()
    dst = load_dst()
    X = build_features(imf)

    data = X.join(dst, how="inner").loc[START:END]

    features = list(X.columns)
    n = len(features)
    left_feats = features[: n // 2]
    right_feats = features[n // 2 :]

    nrows = max(len(left_feats), len(right_feats)) + 1

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=2,
        figsize=(16, 2.2 * nrows),
        sharex=True
    )

    # ---------------- Dst (top row, both columns) ----------------

    for col in [0, 1]:
        axes[0, col].plot(data.index, data["dst"], lw=1.2)
        axes[0, col].axhline(-50, ls=":", alpha=0.5)
        axes[0, col].set_ylabel("Dst (nT)")
        axes[0, col].set_title("Dst")

    # ---------------- Left column features ----------------

    for i, name in enumerate(left_feats, start=1):
        ax = axes[i, 0]
        ax.plot(data.index, data[name], lw=1)
        ax.set_ylabel(name)

        if "frac" in name or name == "shock_like":
            ax.set_ylim(-0.05, 1.05)

    # ---------------- Right column features ----------------

    for i, name in enumerate(right_feats, start=1):
        ax = axes[i, 1]
        ax.plot(data.index, data[name], lw=1)
        ax.set_ylabel(name)

        if "frac" in name or name == "shock_like":
            ax.set_ylim(-0.05, 1.05)

    # Hide empty panels if uneven
    for i in range(len(left_feats) + 1, nrows):
        axes[i, 0].axis("off")
    for i in range(len(right_feats) + 1, nrows):
        axes[i, 1].axis("off")

    axes[-1, 0].set_xlabel("Time (UTC)")
    axes[-1, 1].set_xlabel("Time (UTC)")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

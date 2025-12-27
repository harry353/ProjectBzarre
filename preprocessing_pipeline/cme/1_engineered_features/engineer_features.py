from __future__ import annotations
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve()
for parent in PROJECT_ROOT.parents:
    if (parent / "space_weather_api.py").exists():
        PROJECT_ROOT = parent
        break
else:
    PROJECT_ROOT = PROJECT_ROOT.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from preprocessing_pipeline.utils import write_sqlite_table

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
STAGE_DIR = Path(__file__).resolve().parent

CATALOG_DB = STAGE_DIR.parents[1] / "space_weather.db"
CATALOG_TABLE = "lasco_cme_catalog"

OUTPUT_DB = STAGE_DIR / "cme_hourly_eng.db"
OUTPUT_TABLE = "engineered_features"

# ---------------------------------------------------------------------
# Time grid
# ---------------------------------------------------------------------
START = pd.Timestamp("1998-01-01T00:00:00Z")
END = pd.Timestamp("2025-11-30T23:00:00Z")
HOURLY_INDEX = pd.date_range(START, END, freq="1h", tz="UTC")

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
HALO_MAP = {"N": 0, "P": 1, "H": 2}
AU_KM = 1.496e8
DECAY_TAU_HOURS = 36.0


def _load_cme_catalog() -> pd.DataFrame:
    import sqlite3

    with sqlite3.connect(CATALOG_DB) as conn:
        df = pd.read_sql_query(
            f"SELECT * FROM {CATALOG_TABLE}",
            conn,
            parse_dates=["time_tag"],
        )
    df["time_tag"] = pd.to_datetime(df["time_tag"], utc=True)
    return df.sort_values("time_tag")


# ---------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------
def engineer_cme_features() -> pd.DataFrame:
    cme = _load_cme_catalog()
    cme = cme.set_index("time_tag").sort_index()
    cme = cme[~cme.index.duplicated(keep="last")]

    # --------------------------------------------------------------
    # Precompute CME-to-CME interaction quantities
    # --------------------------------------------------------------
    cme["prev_cme_v_med"] = cme["median_velocity"].shift(1).fillna(0.0)
    cme["delta_last_cme_speed"] = (
        cme["median_velocity"] - cme["prev_cme_v_med"]
    ).fillna(0.0)

    cme["delta_last_cme_width"] = (
        cme["angular_width"] - cme["angular_width"].shift(1)
    ).fillna(0.0)

    cme["last_cme_speed_ratio"] = (
        cme["median_velocity"]
        / np.maximum(cme["prev_cme_v_med"], 1.0)
    ).fillna(0.0)

    cme["cme_overtaking_flag"] = (cme["last_cme_speed_ratio"] > 1.5).astype(int)

    # Strength and shock proxies
    cme["strength"] = cme["median_velocity"] * cme["angular_width"]
    cme["shock_proxy"] = (
        cme["median_velocity"]
        * (cme["velocity_variation"] / np.maximum(cme["median_velocity"], 1.0))
    )

    # Geometry
    pa = np.deg2rad(cme["position_angle"])
    cme["pa_sin"] = np.sin(pa)
    cme["pa_cos"] = np.cos(pa)

    # Severity class
    cme["cme_severity_class"] = pd.cut(
        cme["median_velocity"],
        bins=[-np.inf, 500, 800, 1500, np.inf],
        labels=[0, 1, 2, 3],
    ).astype(int)

    # --------------------------------------------------------------
    # Hourly backbone
    # --------------------------------------------------------------
    hourly = pd.DataFrame(index=HOURLY_INDEX)
    hourly.index.name = "time_tag"

    # --------------------------------------------------------------
    # Event flags
    # --------------------------------------------------------------
    hourly_counts = (
        pd.Series(1, index=cme.index)
        .groupby(pd.Grouper(freq="1h"))
        .sum()
        .reindex(hourly.index, fill_value=0)
    )
    hourly["cme_active_flag"] = (hourly_counts > 0).astype(int)

    # --------------------------------------------------------------
    # Time since / until CME
    # --------------------------------------------------------------
    last_event = cme.index.to_series().reindex(hourly.index, method="ffill")
    next_event = cme.index.to_series().reindex(hourly.index, method="bfill")

    hourly["hours_since_last_cme"] = (
        (hourly.index - last_event).dt.total_seconds() / 3600
    ).fillna(1e6)

    hourly["hours_until_next_cme"] = (
        (next_event - hourly.index).dt.total_seconds() / 3600
    ).fillna(1e6)

    # --------------------------------------------------------------
    # CME rate and strength-weighted rate
    # --------------------------------------------------------------
    def _rolling_sum_np(series: pd.Series, window: int) -> pd.Series:
        values = series.to_numpy(dtype=float)
        cumsum = np.cumsum(values)
        result = cumsum.copy()
        if window < len(values):
            result[window:] = cumsum[window:] - cumsum[:-window]
        return pd.Series(result, index=series.index, dtype=float)

    hourly["cme_count_last_24h"] = _rolling_sum_np(hourly_counts, 24)
    hourly["cme_count_last_72h"] = _rolling_sum_np(hourly_counts, 72)

    strength_series = cme["strength"].groupby(pd.Grouper(freq="1h")).sum()
    strength_series = strength_series.reindex(hourly.index, fill_value=0.0)
    hourly["cme_strength_sum_24h"] = _rolling_sum_np(strength_series, 24)
    hourly["cme_strength_sum_72h"] = _rolling_sum_np(strength_series, 72)

    hourly["cme_cluster_intense_flag"] = (
        (hourly["cme_count_last_72h"] >= 3)
        & (hourly["cme_strength_sum_72h"] > 2e5)
    ).astype(int)

    # --------------------------------------------------------------
    # Carry-forward CME properties
    # --------------------------------------------------------------
    def ff(col, fill=0.0):
        return col.reindex(hourly.index, method="ffill").fillna(fill)

    hourly["last_cme_v_med"] = ff(cme["median_velocity"])
    hourly["last_cme_width"] = ff(cme["angular_width"])
    hourly["last_cme_strength"] = ff(cme["strength"])
    hourly["last_cme_halo_ord"] = ff(cme["halo_class"].map(HALO_MAP), 0).astype(int)

    hourly["last_cme_pa_sin"] = ff(cme["pa_sin"])
    hourly["last_cme_pa_cos"] = ff(cme["pa_cos"])

    hourly["last_cme_fast_flag"] = (hourly["last_cme_v_med"] >= 800).astype(int)
    hourly["last_cme_shock_proxy"] = ff(cme["shock_proxy"])

    hourly["prev_cme_v_med"] = ff(cme["prev_cme_v_med"])
    hourly["last_cme_speed_ratio"] = ff(cme["last_cme_speed_ratio"])
    hourly["cme_overtaking_flag"] = ff(cme["cme_overtaking_flag"], 0).astype(int)
    hourly["delta_last_cme_speed"] = ff(cme["delta_last_cme_speed"])
    hourly["delta_last_cme_width"] = ff(cme["delta_last_cme_width"])

    # --------------------------------------------------------------
    # Earth-facing geometry
    # --------------------------------------------------------------
    earth_alignment = hourly["last_cme_pa_cos"].clip(-1, 1)
    hourly["earth_alignment_score"] = earth_alignment
    hourly["earth_facing_flag"] = (earth_alignment > 0.7).astype(int)
    hourly["effective_width"] = hourly["last_cme_width"] * earth_alignment.clip(0, 1)

    # --------------------------------------------------------------
    # Transit-time heuristics
    # --------------------------------------------------------------
    hourly["cme_arrival_est_hours"] = (
        AU_KM / np.maximum(hourly["last_cme_v_med"], 1.0)
    )

    hourly["hours_until_est_arrival"] = (
        hourly["cme_arrival_est_hours"] - hourly["hours_since_last_cme"]
    )

    hourly["arrival_window_flag"] = (
        hourly["hours_until_est_arrival"].abs() <= 12
    ).astype(int)

    # --------------------------------------------------------------
    # CME influence decay
    # --------------------------------------------------------------
    hourly["cme_influence_exp"] = np.exp(
        -hourly["hours_since_last_cme"] / DECAY_TAU_HOURS
    )

    # --------------------------------------------------------------
    # Severity class carry-forward
    # --------------------------------------------------------------
    hourly["cme_severity_class"] = ff(cme["cme_severity_class"], 0).astype(int)

    # --------------------------------------------------------------
    # Final safety check with diagnostics
    # --------------------------------------------------------------
    if hourly.isna().any().any():
        allowed_mask = hourly.index.year == 1998
        na_counts = hourly.isna().sum()
        offending: dict[str, int] = {}
        for col in hourly.columns:
            mask = hourly[col].isna()
            if not mask.any():
                continue
            disallowed = mask & ~allowed_mask
            if disallowed.any():
                offending[col] = int(disallowed.sum())
            else:
                # only early warm-up NaNs: fill with zeros and continue
                fill_value = 0.0
                hourly.loc[mask, col] = fill_value
        if offending:
            details = ", ".join(f"{col} ({count})" for col, count in offending.items())
            raise RuntimeError(f"NaNs detected in CME feature table: {details}")

    write_sqlite_table(hourly, OUTPUT_DB, OUTPUT_TABLE)

    print(f"[OK] CME engineered features written to {OUTPUT_DB}")
    return hourly


def main() -> None:
    engineer_cme_features()


if __name__ == "__main__":
    main()

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

SSC_DB = PIPELINE_ROOT / "features_targets" / "full_storm_label" / "full_storm_labels.db"

DST_DB = PIPELINE_ROOT / "dst" / "1_averaging" / "dst_aver.db"
KP_DB = PIPELINE_ROOT / "kp_index" / "1_averaging" / "kp_index_aver.db"

MODEL_PATH  = PIPELINE_ROOT / "features_targets" / "disturbed_label" / "stage1_model.joblib"
SCALER_PATH = PIPELINE_ROOT / "features_targets" / "disturbed_label" / "stage1_scaler.joblib"

YEAR = 2024
P_THRESHOLD = 0.45
PLOT_ONLY_ABOVE_THRESH = False
LEAD_HOURS = 8

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


def load_storm_onsets() -> list[pd.Timestamp]:
    frames = []
    with sqlite3.connect(SSC_DB) as conn:
        for t in [
            "storm_full_storm_train",
            "storm_full_storm_validation",
            "storm_full_storm_test",
        ]:
            if not pd.read_sql(
                "SELECT name FROM sqlite_master WHERE name=?",
                conn,
                params=(t,),
            ).empty:
                frames.append(
                    pd.read_sql(
                        f"SELECT timestamp, storm_flag FROM {t}",
                        conn,
                        parse_dates=["timestamp"],
                    )
                )

    full_storm = pd.concat(frames).drop_duplicates("timestamp")
    full_storm = full_storm.set_index("timestamp").sort_index()
    full_storm.index = ensure_utc_index(full_storm.index)

    return list(full_storm.index[full_storm["storm_flag"] == 1])


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


def load_kp() -> pd.DataFrame:
    with sqlite3.connect(KP_DB) as conn:
        df = pd.read_sql(
            "SELECT time_tag AS t, kp_index FROM hourly_data",
            conn,
            parse_dates=["t"],
        )
    df = df.set_index("t").sort_index()
    df.index = ensure_utc_index(df.index)
    return df

# ---------------------------------------------------------------------
# Feature engineering (must match training)
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


    return X.dropna()

# ---------------------------------------------------------------------
# Disturbed spans
# ---------------------------------------------------------------------

def disturbed_spans(
    prob: pd.Series, threshold: float
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    active = prob >= threshold
    run_id = active.ne(active.shift(fill_value=False)).cumsum()

    spans = []
    for _, block in active.groupby(run_id):
        if not block.iloc[0]:
            continue
        start = block.index[0]
        end = block.index[-1] + pd.Timedelta(hours=1)
        spans.append((start, end))

    return spans

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    imf = load_imf()
    dst = load_dst()
    kp = load_kp()

    X = build_features(imf)

    scaler = joblib.load(SCALER_PATH)
    model = joblib.load(MODEL_PATH)

    Xs = scaler.transform(X)
    probs = pd.Series(
        model.predict_proba(Xs)[:, 1],
        index=X.index,
        name="p",
    )

    start = pd.Timestamp(f"{YEAR}-01-01", tz="UTC")
    end = start + pd.DateOffset(years=1)

    dst = dst.loc[start:end]
    kp = kp.loc[start:end]
    probs = probs.loc[start:end]


    # -----------------------------------------------------------------
    # Plot
    # -----------------------------------------------------------------

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 6), sharex=True, height_ratios=[2, 1]
    )

    ax1.plot(dst.index, dst["dst"], color="tab:blue", lw=1)
    ax2.plot(kp.index, kp["kp_index"], color="tab:gray", lw=1)

    norm = colors.Normalize(vmin=0.0, vmax=1.0)
    cmap = cm.get_cmap("YlOrRd")
    for t, p in probs.items():
        if PLOT_ONLY_ABOVE_THRESH and p < P_THRESHOLD:
            continue
        color = cmap(norm(p))
        ax1.axvspan(t, t + pd.Timedelta(hours=1), color=color, alpha=0.35, lw=0)
        ax2.axvspan(t, t + pd.Timedelta(hours=1), color=color, alpha=0.35, lw=0)

    ax1.axhline(-50, ls=":", color="black", alpha=0.4)
    ax1.axhline(0, ls=":", color="black", alpha=0.3)
    ax2.axhline(5, ls=":", color="black", alpha=0.4)
    ax2.axhline(0, ls=":", color="black", alpha=0.3)

    ax1.set_ylabel("Dst (nT)")
    ax2.set_ylabel("Kp")
    ax2.set_xlabel("Time")

    ax1.set_title(f"Stage-1 ML probabilities – {YEAR}")
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

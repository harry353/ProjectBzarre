from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------
# Project discovery
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve()
for parent in PROJECT_ROOT.parents:
    if (parent / "space_weather_api.py").exists():
        PROJECT_ROOT = parent
        break
else:
    PROJECT_ROOT = PROJECT_ROOT.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

PIPELINE_ROOT = PROJECT_ROOT / "preprocessing_pipeline"

DISTURBED_DB = PIPELINE_ROOT / "features_targets" / "disturbed_label" / "disturbed_labels.db"
DISTURBED_TABLES = {
    "train": "disturbed_train",
    "val": "disturbed_validation",
    "test": "disturbed_test",
}

SSC_DB = PIPELINE_ROOT / "features_targets" / "full_storm_label" / "full_storm_labels.db"
SSC_TABLES = [
    "storm_full_storm_train",
    "storm_full_storm_validation",
    "storm_full_storm_test",
]

OUTPUT_DB = PIPELINE_ROOT / "features_targets" / "stage2_labels" / "stage2_labels.db"

STAGE2_TABLES = {
    "train": "stage2_train",
    "val": "stage2_validation",
    "test": "stage2_test",
}

STAGE2_HORIZON_HOURS = 8

# ---------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------

def _ensure_utc(ts: pd.Series) -> pd.Series:
    return ts.dt.tz_localize("UTC") if ts.dt.tz is None else ts.dt.tz_convert("UTC")


def load_disturbed(split: str) -> pd.DataFrame:
    with sqlite3.connect(DISTURBED_DB) as conn:
        df = pd.read_sql_query(
            f"""
            SELECT time_tag AS timestamp, disturbed_flag
            FROM {DISTURBED_TABLES[split]}
            """,
            conn,
            parse_dates=["timestamp"],
        )
    df["timestamp"] = _ensure_utc(df["timestamp"])
    return df.set_index("timestamp").sort_index()


def load_storm_onsets() -> list[pd.Timestamp]:
    frames = []
    with sqlite3.connect(SSC_DB) as conn:
        for table in SSC_TABLES:
            if not pd.read_sql_query(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                conn,
                params=(table,),
            ).empty:
                frame = pd.read_sql_query(
                    f"""
                    SELECT timestamp, storm_flag
                    FROM {table}
                    """,
                    conn,
                    parse_dates=["timestamp"],
                )
                frames.append(frame)

    if not frames:
        raise RuntimeError("No SSC tables found")

    df = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates("timestamp")
        .sort_values("timestamp")
    )
    df["timestamp"] = _ensure_utc(df["timestamp"])
    df = df.set_index("timestamp")

    active = df["storm_flag"] == 1
    run_id = active.ne(active.shift(fill_value=False)).cumsum()

    onsets = []
    for run, flag in active.groupby(run_id):
        if flag.iloc[0]:
            onsets.append(df.index[run_id == run][0])

    return onsets


# ---------------------------------------------------------------------
# Stage-2 label logic
# ---------------------------------------------------------------------

def build_stage2_labels(
    disturbed_index: pd.DatetimeIndex,
    storm_onsets: list[pd.Timestamp],
    horizon_hours: int,
) -> pd.Series:
    storm_onsets = pd.Series(storm_onsets)
    labels = []

    for t in disturbed_index:
        window_end = t + pd.Timedelta(hours=horizon_hours)
        hit = ((storm_onsets > t) & (storm_onsets <= window_end)).any()
        labels.append(int(hit))

    return pd.Series(
        labels,
        index=disturbed_index,
        name=f"storm_within_{horizon_hours}h",
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    storm_onsets = load_storm_onsets()

    OUTPUT_DB.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(OUTPUT_DB) as conn:
        for split in ("train", "val", "test"):
            disturbed = load_disturbed(split)

            disturbed = disturbed[disturbed["disturbed_flag"] == 1]
            if disturbed.empty:
                raise RuntimeError(f"No disturbed rows in {split}")

            y = build_stage2_labels(
                disturbed_index=disturbed.index,
                storm_onsets=storm_onsets,
                horizon_hours=STAGE2_HORIZON_HOURS,
            )

            out = disturbed.copy()
            out[y.name] = y.values

            out.reset_index().to_sql(
                STAGE2_TABLES[split],
                conn,
                if_exists="replace",
                index=False,
            )

            pos_rate = y.mean()
            print(
                f"[OK] {split}: {len(out):,} rows | "
                f"positive rate = {pos_rate:.3f}"
            )

    print("\nStage-2 label construction complete.")


if __name__ == "__main__":
    main()


from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


STAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = STAGE_DIR
for parent in STAGE_DIR.parents:
    if (parent / "space_weather_api.py").exists():
        PROJECT_ROOT = parent
        break
else:
    PROJECT_ROOT = STAGE_DIR.parent


DST_DB = PROJECT_ROOT / "preprocessing_pipeline" / "space_weather.db"
DST_TABLE = "dst_index"
LABEL_DB = STAGE_DIR / "dst_regression_labels.db"
LABEL_TABLES = (
    "dst_regression_train",
    "dst_regression_validation",
    "dst_regression_test",
)
FORECAST_HORIZONS_H = range(1, 9)


def _ensure_utc(series: pd.Series) -> pd.Series:
    return series.dt.tz_localize("UTC") if series.dt.tz is None else series.dt.tz_convert("UTC")


def _pick_dst_table(conn: sqlite3.Connection, db_path: Path) -> str:
    tables = [row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]
    if not tables:
        raise RuntimeError(f"No tables found in {db_path}")
    if DST_TABLE in tables:
        return DST_TABLE
    if len(tables) == 1:
        return tables[0]
    for table in tables:
        cols = [row[1] for row in conn.execute(f"PRAGMA table_info({table})")]
        if "dst" in cols:
            return table
    raise RuntimeError(f"Could not find DST table in {db_path}. Available: {tables}")


def _load_dst() -> pd.Series:
    with sqlite3.connect(DST_DB) as conn:
        table = _pick_dst_table(conn, DST_DB)
        df = pd.read_sql_query(
            f"SELECT * FROM {table}",
            conn,
            parse_dates=["timestamp", "time_tag", "date"],
        )
    if df.empty:
        raise RuntimeError(f"No DST rows found in {DST_DB}:{table}")
    time_col = None
    for candidate in ("time_tag", "timestamp", "date"):
        if candidate in df.columns:
            time_col = candidate
            break
    if time_col is None:
        raise RuntimeError(f"No timestamp column found in {DST_DB}:{table}")
    df = df.rename(columns={time_col: "timestamp"})
    df["timestamp"] = _ensure_utc(df["timestamp"])
    if "dst" not in df.columns:
        raise RuntimeError(f"No dst column found in {DST_DB}:{table}")
    return df.set_index("timestamp")["dst"].astype(float).sort_index()


def _load_labels() -> pd.DataFrame:
    if not LABEL_DB.exists():
        raise FileNotFoundError(f"Label DB not found at {LABEL_DB}; run build_dst_regression_labels.py first.")

    frames: list[pd.DataFrame] = []
    with sqlite3.connect(LABEL_DB) as conn:
        for table in LABEL_TABLES:
            try:
                df = pd.read_sql_query(f"SELECT * FROM {table}", conn, parse_dates=["timestamp"])
            except Exception:
                continue
            if not df.empty:
                frames.append(df)

    if not frames:
        raise RuntimeError(f"No label tables found in {LABEL_DB}; expected one of {LABEL_TABLES}.")

    labels = pd.concat(frames, ignore_index=True)
    labels["timestamp"] = _ensure_utc(labels["timestamp"])
    labels = labels.set_index("timestamp").sort_index()
    return labels


def _plot_year(year: int) -> None:
    dst = _load_dst()
    labels = _load_labels()

    combined = labels.join(dst.rename("dst"), how="inner")
    plot_df = combined[combined.index.year == year]
    if plot_df.empty:
        raise RuntimeError(f"No data available for year {year}.")

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(plot_df.index, plot_df["dst"], label="dst (observed)", color="black", linewidth=1.2)

    for h in FORECAST_HORIZONS_H:
        col = f"h{h}"
        if col not in plot_df.columns:
            continue
        ax.plot(plot_df.index, plot_df[col], label=col, linewidth=0.9, alpha=0.7)

    ax.set_title(f"DST and horizons (hourly) — {year}")
    ax.set_xlabel("Timestamp (UTC)")
    ax.set_ylabel("DST index")
    ax.legend(ncol=3, fontsize="small")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.autofmt_xdate()

    fig.tight_layout()
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot DST and regression horizons for a given year.")
    parser.add_argument("--year", type=int, default=None, help="Year to plot, e.g., 2015")
    args = parser.parse_args()

    year = args.year
    if year is None:
        labels = _load_labels()
        year = int(labels.index.year.max())
    _plot_year(year)


if __name__ == "__main__":
    main()

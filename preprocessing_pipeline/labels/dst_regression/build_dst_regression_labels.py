from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

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

OUTPUT_DB = STAGE_DIR / "dst_regression_labels.db"
TRAIN_TABLE = "dst_regression_train"
VAL_TABLE = "dst_regression_validation"
TEST_TABLE = "dst_regression_test"

FORECAST_HORIZONS_H = range(1, 9)

TRAIN_START = pd.Timestamp("1999-01-01T00:00:00Z")
TRAIN_END = pd.Timestamp("2016-12-31T23:59:59Z")
VAL_START = pd.Timestamp("2017-01-01T00:00:00Z")
VAL_END = pd.Timestamp("2020-12-31T23:59:59Z")
TEST_START = pd.Timestamp("2021-01-01T00:00:00Z")
TEST_END = pd.Timestamp("2025-11-30T23:59:59Z")


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


def _load_dst(db_path: Path, table_name: str | None = None) -> pd.Series:
    with sqlite3.connect(db_path) as conn:
        table = table_name or _pick_dst_table(conn, db_path)
        df = pd.read_sql_query(
            f"SELECT * FROM {table}",
            conn,
            parse_dates=["timestamp", "time_tag", "date"],
        )
    if df.empty:
        raise RuntimeError(f"No rows found in {db_path}:{table}")
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
        raise RuntimeError(f"No dst column found in {db_path}:{table}")
    return df.set_index("timestamp")["dst"].astype(float).sort_index()


def _candidate_dst_dbs() -> list[Path]:
    preferred = [
        PROJECT_ROOT / "preprocessing_pipeline" / "space_weather.db",
        PROJECT_ROOT / "preprocessing_pipeline" / "dst" / "1_averaging" / "dst_aver.db",
        PROJECT_ROOT / "preprocessing_pipeline" / "dst" / "4_imputation" / "dst_aver_filt_imp.db",
        PROJECT_ROOT / "preprocessing_pipeline" / "dst" / "5_train_test_split" / "dst_imputed_split.db",
        PROJECT_ROOT / "preprocessing_pipeline" / "dst" / "dst_fin.db",
    ]
    seen = set()
    out: list[Path] = []
    for path in preferred:
        if path.exists() and path not in seen:
            out.append(path)
            seen.add(path)
    for path in (PROJECT_ROOT / "preprocessing_pipeline" / "dst").rglob("*.db"):
        if path not in seen:
            out.append(path)
            seen.add(path)
    return out


def _resolve_dst_source(db_path: Path | None) -> Path:
    if db_path is not None:
        if not db_path.exists():
            raise FileNotFoundError(f"DST db not found: {db_path}")
        return db_path

    for path in _candidate_dst_dbs():
        try:
            with sqlite3.connect(path) as conn:
                _pick_dst_table(conn, path)
            return path
        except Exception:
            continue
    raise RuntimeError(
        "No usable DST database found. Run preprocessing_pipeline/dst/1_averaging/build_hourly.py "
        "or pass --dst-db."
    )


def _build_horizon_targets(dst: pd.Series) -> pd.DataFrame:
    labels = pd.DataFrame(index=dst.index)
    for h in FORECAST_HORIZONS_H:
        labels[f"h{h}"] = dst.shift(-h)
    labels = labels.dropna()
    return labels.reset_index().rename(columns={"index": "timestamp"})


def _split(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    return df[(df["timestamp"] >= start) & (df["timestamp"] <= end)].copy()


def build_dst_regression_labels(db_path: Path | None = None, table_name: str | None = None) -> None:
    dst_db = _resolve_dst_source(db_path)
    dst = _load_dst(dst_db, table_name)
    labels = _build_horizon_targets(dst)

    train = _split(labels, TRAIN_START, TRAIN_END)
    val = _split(labels, VAL_START, VAL_END)
    test = _split(labels, TEST_START, TEST_END)

    OUTPUT_DB.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(OUTPUT_DB) as conn:
        train.to_sql(TRAIN_TABLE, conn, if_exists="replace", index=False)
        val.to_sql(VAL_TABLE, conn, if_exists="replace", index=False)
        test.to_sql(TEST_TABLE, conn, if_exists="replace", index=False)

    print("[OK] DST regression labels written to", OUTPUT_DB)
    print(f"     Train / Val / Test rows: {len(train):,} / {len(val):,} / {len(test):,}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build DST regression labels for horizons 1-8.")
    parser.add_argument("--dst-db", type=Path, default=None, help="Path to DST sqlite DB.")
    parser.add_argument("--dst-table", type=str, default=None, help="DST table name override.")
    args = parser.parse_args()

    build_dst_regression_labels(args.dst_db, args.dst_table)


if __name__ == "__main__":
    main()

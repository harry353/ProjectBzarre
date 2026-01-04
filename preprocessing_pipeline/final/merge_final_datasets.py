from __future__ import annotations

import sqlite3
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve()
for parent in PROJECT_ROOT.parents:
    if (parent / "space_weather_api.py").exists():
        PROJECT_ROOT = parent
        break
else:
    PROJECT_ROOT = PROJECT_ROOT.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

FINAL_DIR = PROJECT_ROOT / "preprocessing_pipeline"

SOURCES = [
    {
        "name": "dst",
        "db": FINAL_DIR / "dst" / "dst_fin.db",
        "tables": {
            "train": "dst_train",
            "validation": "dst_validation",
            "test": "dst_test",
        },
    },
    {
        "name": "kp",
        "db": FINAL_DIR / "kp_index" / "kp_fin.db",
        "tables": {
            "train": "kp_train",
            "validation": "kp_validation",
            "test": "kp_test",
        },
    },
    {
        "name": "sunspot",
        "db": FINAL_DIR / "sunspot_number" / "sunspot_number_fin.db",
        "tables": {
            "train": "sunspot_train",
            "validation": "sunspot_validation",
            "test": "sunspot_test",
        },
    },
#    {
#        "name": "flares",
#        "db": FINAL_DIR / "flares" / "flare_fin.db",
#        "tables": {
#            "train": "flare_train",
#            "validation": "flare_validation",
#            "test": "flare_test",
#        },
#    },
    {
        "name": "cme",
        "db": FINAL_DIR / "cme" / "cme_fin.db",
        "tables": {
            "train": "cme_train",
            "validation": "cme_validation",
            "test": "cme_test",
        },
    },
    {
        "name": "imf_solar_wind",
        "db": FINAL_DIR / "imf_solar_wind" / "imf_solar_wind_fin.db",
        "tables": {
            "train": "imf_solar_wind_train",
            "validation": "imf_solar_wind_validation",
            "test": "imf_solar_wind_test",
        },
    },
    # {
    #     "name": "xray_flux",
    #     "db": FINAL_DIR / "xray_flux" / "xray_flux_fin.db",
    #     "tables": {
    #         "train": "xray_flux_train",
    #         "validation": "xray_flux_validation",
    #         "test": "xray_flux_test",
    #     },
    # },
    {
        "name": "radio_flux",
        "db": FINAL_DIR / "radio_flux" / "radio_flux_fin.db",
        "tables": {
            "train": "radio_flux_train",
            "validation": "radio_flux_validation",
            "test": "radio_flux_test",
        },
    },
    {
        "name": "full_storm_labels",
        "db": FINAL_DIR / "labels" / "full_storm_label" / "full_storm_labels.db",
        "tables": {
            "train": "storm_full_storm_train",
            "validation": "storm_full_storm_validation",
            "test": "storm_full_storm_test",
        },
        "columns": [
            "storm_flag_h1",
            "storm_flag_h2",
            "storm_flag_h3",
            "storm_flag_h4",
            "storm_flag_h5",
            "storm_flag_h6",
            "storm_flag_h7",
            "storm_flag_h8",
        ],
    },
]

OUTPUT_DB = FINAL_DIR / "final" / "all_preprocessed_sources.db"

CHUNK_SIZE = 50_000


def _detect_time_col(conn: sqlite3.Connection, table: str) -> str:
    cols = [row[1] for row in conn.execute(f"PRAGMA table_info({table})")]
    for candidate in ("timestamp", "time_tag", "date"):
        if candidate in cols:
            return candidate
    raise RuntimeError(f"Table '{table}' has no timestamp column.")


def _normalize_time(series: pd.Series, time_col: str) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce")
    if time_col == "date":
        if parsed.dt.tz is not None:
            parsed = parsed.dt.tz_convert(None)
    else:
        if parsed.dt.tz is None:
            parsed = parsed.dt.tz_localize("UTC")
        else:
            parsed = parsed.dt.tz_convert("UTC")
        parsed = parsed.dt.tz_localize(None)
    return parsed


def _write_temp_table(
    out_conn: sqlite3.Connection,
    db_path: Path,
    table: str,
    tmp_table: str,
    prefix: str,
    keep_cols: Optional[Iterable[str]] = None,
) -> List[str]:
    with sqlite3.connect(db_path) as src_conn:
        time_col = _detect_time_col(src_conn, table)
        query = f"SELECT * FROM {table}"
        first = True
        for chunk in pd.read_sql_query(query, src_conn, chunksize=CHUNK_SIZE):
            if time_col not in chunk.columns:
                raise RuntimeError(f"Missing time column '{time_col}' in {db_path}:{table}")
            if keep_cols is not None:
                keep = [time_col, *[c for c in keep_cols if c != time_col]]
                missing = [c for c in keep if c not in chunk.columns]
                if missing:
                    raise RuntimeError(
                        f"Missing expected columns {missing} in {db_path}:{table}"
                    )
                chunk = chunk[keep]
            parsed = _normalize_time(chunk[time_col], time_col)
            chunk = chunk.assign(time_key=parsed)
            chunk = chunk.dropna(subset=["time_key"])
            chunk = chunk.drop(columns=[time_col])
            chunk = chunk.rename(columns={c: f"{prefix}_{c}" for c in chunk.columns if c != "time_key"})
            chunk["time_key"] = chunk["time_key"].dt.strftime("%Y-%m-%d %H:%M:%S")
            if first:
                chunk.to_sql(tmp_table, out_conn, if_exists="replace", index=False)
                first = False
            else:
                chunk.to_sql(tmp_table, out_conn, if_exists="append", index=False)
    out_conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{tmp_table}_time ON {tmp_table}(time_key)")
    cols = [row[1] for row in out_conn.execute(f"PRAGMA table_info({tmp_table})")]
    return cols


def main() -> None:
    OUTPUT_DB.parent.mkdir(parents=True, exist_ok=True)

    splits = ("train", "validation", "test")
    with sqlite3.connect(OUTPUT_DB) as out_conn:
        for split in splits:
            temp_tables: List[str] = []
            temp_columns: Dict[str, List[str]] = {}

            for cfg in SOURCES:
                db_path = cfg["db"]
                if not db_path.exists():
                    raise FileNotFoundError(f"Required final DB missing: {db_path}")

                src_table = cfg["tables"][split]
                tmp_table = f"tmp_{cfg['name']}_{split}"
                temp_tables.append(tmp_table)
                temp_columns[tmp_table] = _write_temp_table(
                    out_conn,
                    db_path,
                    src_table,
                    tmp_table,
                    cfg["name"],
                    cfg.get("columns"),
                )

            if not temp_tables:
                raise RuntimeError(f"No data merged for split '{split}'.")

            select_cols: List[str] = []
            join_sql = f"FROM {temp_tables[0]} t0"
            first_cols = [c for c in temp_columns[temp_tables[0]] if c != "time_key"]
            select_cols.append("t0.time_key AS timestamp")
            select_cols.extend([f"t0.{c}" for c in first_cols])

            for idx, tmp in enumerate(temp_tables[1:], start=1):
                alias = f"t{idx}"
                join_sql += f" INNER JOIN {tmp} {alias} USING (time_key)"
                cols = [c for c in temp_columns[tmp] if c != "time_key"]
                select_cols.extend([f"{alias}.{c}" for c in cols])

            table_name = f"merged_{split}"
            out_conn.execute(f"DROP TABLE IF EXISTS {table_name}")
            out_conn.execute(
                f"CREATE TABLE {table_name} AS SELECT {', '.join(select_cols)} {join_sql}"
            )
            row_count = out_conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
            print(f"[OK] Created {table_name} ({row_count:,} rows)")

            for tmp in temp_tables:
                out_conn.execute(f"DROP TABLE IF EXISTS {tmp}")

    print(f"[OK] All preprocessed sources saved to {OUTPUT_DB}")


if __name__ == "__main__":
    main()

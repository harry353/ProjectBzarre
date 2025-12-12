from __future__ import annotations

import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

PREPROCESSING_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PREPROCESSING_DIR
PYTHON_EXE = Path(sys.executable)

FINAL_DB = PREPROCESSING_DIR / "space_weather_preprocessed.db"
FINAL_PARQUET = PREPROCESSING_DIR / "space_weather_preprocessed.parquet"

SOURCES: List[Dict[str, Path | str]] = [
    {"name": "ae", "script": PREPROCESSING_DIR / "ae" / "make_ae_hourly.py", "db": PREPROCESSING_DIR / "ae" / "ae_hourly.db"},
    {"name": "cme", "script": PREPROCESSING_DIR / "cme" / "make_cme_hourly_features.py", "db": PREPROCESSING_DIR / "cme" / "cme_hourly.db"},
    {"name": "dst", "script": PREPROCESSING_DIR / "dst" / "make_dst_hourly.py", "db": PREPROCESSING_DIR / "dst" / "dst_hourly.db"},
    {"name": "flares", "script": PREPROCESSING_DIR / "flares" / "make_flares_hourly_features.py", "db": PREPROCESSING_DIR / "flares" / "flares_hourly.db"},
    {"name": "imf", "script": PREPROCESSING_DIR / "imf" / "make_imf_hourly.py", "db": PREPROCESSING_DIR / "imf" / "imf_hourly.db"},
    {"name": "kp_index", "script": PREPROCESSING_DIR / "kp_index" / "make_kp_index_hourly.py", "db": PREPROCESSING_DIR / "kp_index" / "kp_index_hourly.db"},
    {"name": "radio_flux", "script": PREPROCESSING_DIR / "radio_flux" / "make_radio_flux_hourly.py", "db": PREPROCESSING_DIR / "radio_flux" / "radio_flux_hourly.db"},
    {"name": "solar_wind", "script": PREPROCESSING_DIR / "solar_wind" / "make_solar_wind_hourly.py", "db": PREPROCESSING_DIR / "solar_wind" / "solar_wind_hourly.db"},
    {"name": "sunspot_number", "script": PREPROCESSING_DIR / "sunspot_number" / "make_sunspot_hourly.py", "db": PREPROCESSING_DIR / "sunspot_number" / "sunspot_hourly.db"},
    {"name": "supermag", "script": PREPROCESSING_DIR / "supermag" / "make_supermag_hourly.py", "db": PREPROCESSING_DIR / "supermag" / "supermag_hourly.db"},
    {"name": "sw_comp", "script": PREPROCESSING_DIR / "sw_comp" / "make_sw_comp_hourly.py", "db": PREPROCESSING_DIR / "sw_comp" / "sw_comp_hourly.db"},
    {"name": "xray_flux", "script": PREPROCESSING_DIR / "xray_flux" / "make_xray_flux_hourly.py", "db": PREPROCESSING_DIR / "xray_flux" / "xray_flux_hourly.db"},
]


def run_source_scripts() -> List[Path]:
    produced_dbs: List[Path] = []
    for source in SOURCES:
        script_path: Path = source["script"]  # type: ignore[assignment]
        db_path: Path = source["db"]  # type: ignore[assignment]
        if not script_path.exists():
            print(f"[WARN] Hourly script missing for {source['name']}: {script_path}")
            continue
        print(f"[INFO] Running hourly preprocessing for {source['name']}...")
        result = subprocess.run([PYTHON_EXE, str(script_path)], capture_output=True, text=True)
        if result.returncode != 0:
            print(result.stdout)
            print(result.stderr, file=sys.stderr)
            raise RuntimeError(f"Hourly preprocessing failed for {source['name']}.")
        if db_path.exists():
            produced_dbs.append(db_path)
        else:
            print(f"[WARN] Expected DB not found for {source['name']}: {db_path}")
    return produced_dbs


def build_master_dataframe(db_paths: List[Path]) -> pd.DataFrame:
    frames = []
    used_columns = set()

    for db_path in db_paths:
        with sqlite3.connect(db_path) as conn:
            tables = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';"
            ).fetchall()
            for (table_name,) in tables:
                df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
                if "timestamp" not in df.columns:
                    print(f"[WARN] Table '{table_name}' in {db_path.name} lacks a 'timestamp' column. Skipping.")
                    continue
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
                df = df.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()
                df = _ensure_unique_columns(df, table_name, used_columns)
                frames.append(df)
                print(f"[INFO] Loaded table '{table_name}' from {db_path.name}.")

    if not frames:
        return pd.DataFrame()

    earliest_end = min(frame.index.max() for frame in frames if not frame.empty)
    master = pd.concat(frames, axis=1, join="outer").sort_index()
    if earliest_end is not None:
        master = master.loc[:earliest_end]
    return master


def write_master_outputs(df: pd.DataFrame) -> None:
    df = df.copy()
    if df.empty:
        raise ValueError("Master dataframe is empty.")

    payload = df.reset_index().rename(columns={"index": "timestamp"})

    if FINAL_DB.exists():
        FINAL_DB.unlink()
    with sqlite3.connect(FINAL_DB) as conn:
        payload.to_sql("space_weather_preprocessed", conn, if_exists="replace", index=False)
    print(f"[OK] Master SQLite database created at {FINAL_DB}")

    if FINAL_PARQUET.exists():
        FINAL_PARQUET.unlink()
    payload.to_parquet(FINAL_PARQUET, index=False)
    print(f"[OK] Exported master dataset to {FINAL_PARQUET}")


def _ensure_unique_columns(df: pd.DataFrame, table_name: str, used_columns: set[str]) -> pd.DataFrame:
    rename_map = {}
    for column in df.columns:
        if column == "timestamp":
            continue
        new_name = column
        if new_name in used_columns:
            base = f"{table_name}_{column}"
            suffix = 1
            candidate = base
            while candidate in used_columns:
                candidate = f"{base}_{suffix}"
                suffix += 1
            new_name = candidate
        used_columns.add(new_name)
        rename_map[column] = new_name
    return df.rename(columns=rename_map)


def cleanup(db_paths: List[Path]) -> None:
    for path in db_paths:
        if path.exists():
            path.unlink()
            print(f"[INFO] Removed intermediate DB {path}")


def main() -> None:
    produced_dbs = run_source_scripts()
    if not produced_dbs:
        print("[WARN] No hourly databases were produced. Aborting master preprocessing.")
        return
    master_df = build_master_dataframe(produced_dbs)
    if master_df.empty:
        print("[WARN] Master dataframe is empty. Aborting master preprocessing.")
        return
    write_master_outputs(master_df)
    cleanup(produced_dbs)


if __name__ == "__main__":
    main()

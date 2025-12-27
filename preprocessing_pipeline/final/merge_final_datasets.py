from __future__ import annotations

import sqlite3
import sys
from pathlib import Path
from typing import Dict, Iterable

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
        "db": FINAL_DIR / "kp_index" / "kp_index_fin.db",
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
    {
        "name": "flares",
        "db": FINAL_DIR / "flares" / "flare_fin.db",
        "tables": {
            "train": "flare_train",
            "validation": "flare_validation",
            "test": "flare_test",
        },
    },
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
    {
        "name": "xray_flux",
        "db": FINAL_DIR / "xray_flux" / "xray_flux_fin.db",
        "tables": {
            "train": "xray_flux_train",
            "validation": "xray_flux_validation",
            "test": "xray_flux_test",
        },
    },
    {
        "name": "storm_labels",
        "db": FINAL_DIR / "features_targets" / "storm_labels.db",
        "tables": {
            "train": "severity_train",
            "validation": "severity_validation",
            "test": "severity_test",
        },
    },
    {
        "name": "ssc_labels",
        "db": FINAL_DIR / "features_targets" / "storm_labels.db",
        "tables": {
            "train": "ssc_train",
            "validation": "ssc_validation",
            "test": "ssc_test",
        },
    },
    {
        "name": "main_phase_labels",
        "db": FINAL_DIR / "features_targets" / "storm_labels.db",
        "tables": {
            "train": "main_phase_train",
            "validation": "main_phase_validation",
            "test": "main_phase_test",
        },
    },
]

OUTPUT_DB = FINAL_DIR / "final" / "all_sources_intersection.db"


def _load_table(db_path: Path, table: str) -> pd.DataFrame:
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(f"SELECT * FROM {table}", conn)

    for candidate in ("timestamp", "time_tag"):
        if candidate in df.columns:
            time_col = candidate
            break
    else:
        raise RuntimeError(f"Table '{table}' in {db_path} has no timestamp column.")

    df[time_col] = pd.to_datetime(df[time_col], utc=True, errors="coerce")
    df = df.dropna(subset=[time_col])
    df = df.set_index(time_col).sort_index()
    return df


def _intersect_and_merge(frames: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    common_index = None
    for df in frames.values():
        common_index = df.index if common_index is None else common_index.intersection(df.index)

    if common_index is None or common_index.empty:
        raise RuntimeError("No overlapping timestamps found across sources.")

    aligned_frames = [df.loc[common_index] for df in frames.values()]
    combined = pd.concat(aligned_frames, axis=1)
    combined.index.name = "timestamp"
    combined = combined.reset_index()
    return combined


def main() -> None:
    OUTPUT_DB.parent.mkdir(parents=True, exist_ok=True)

    splits = ("train", "validation", "test")
    with sqlite3.connect(OUTPUT_DB) as out_conn:
        for split in splits:
            frames: Dict[str, pd.DataFrame] = {}
            for cfg in SOURCES:
                db_path = cfg["db"]
                table = cfg["tables"][split]
                if not db_path.exists():
                    raise FileNotFoundError(f"Required final DB missing: {db_path}")
                frames[cfg["name"]] = _load_table(db_path, table)

            combined = _intersect_and_merge(frames)
            table_name = f"combined_{split}"
            combined.to_sql(table_name, out_conn, if_exists="replace", index=False)
            print(f"[OK] Wrote {table_name} with {len(combined):,} rows")

    print(f"[OK] Combined dataset saved to {OUTPUT_DB}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
COMBINED_DB = BASE_DIR / "daily_inference_vector.db"

# Set to a date to fetch that day's 00:00–23:59 UTC, or None for last 24h.
FETCH_DATE: date | None = date(2025, 12, 20)

TABLE_RENAMES = {
    "solar_wind_plasma": "dscovr_f1m",
    "solar_wind_mag": "dscovr_m1m",
    "dst_recent": "dst_index",
    "kp_index_recent": "kp_index",
    "cactus_cme": "lasco_cme_catalog",
    "radio_flux_recent": "radio_flux",
    "sunspot_number_recent": "sunspot_numbers",
    "xray_flux": "xray_flux",
}

SCRIPTS = [
    BASE_DIR / "fetch_goes_xrays.py",
    BASE_DIR / "fetch_cactus_cme.py",
    BASE_DIR / "fetch_dst_recent.py",
    BASE_DIR / "fetch_radio_flux_recent.py",
    BASE_DIR / "fetch_kp_index_recent.py",
    BASE_DIR / "fetch_sunspot_number_recent.py",
    BASE_DIR / "fetch_solar_wind_mag.py",
    BASE_DIR / "fetch_solar_wind_plasma.py",
]


def _run_script(script: Path) -> None:
    if not script.exists():
        raise FileNotFoundError(f"Missing fetch script: {script}")
    env = os.environ.copy()
    if FETCH_DATE is not None:
        env["FETCH_DATE"] = FETCH_DATE.isoformat()
    else:
        env.pop("FETCH_DATE", None)
    subprocess.run([sys.executable, str(script)], check=True, env=env)


def _collect_db_files() -> list[Path]:
    return sorted(
        p
        for p in BASE_DIR.glob("*.db")
        if p.name != COMBINED_DB.name
    )


def _merge_db_files(db_files: list[Path]) -> None:
    COMBINED_DB.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(COMBINED_DB) as out_conn:
        for db_path in db_files:
            with sqlite3.connect(db_path) as in_conn:
                tables = [
                    row[0]
                    for row in in_conn.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    )
                ]
                for table in tables:
                    target_table = TABLE_RENAMES.get(table, table)
                    try:
                        df = pd.read_sql_query(f"SELECT * FROM {table}", in_conn)
                    except sqlite3.Error:
                        continue
                    if df.empty:
                        continue
                    if target_table == "dscovr_m1m":
                        drop_cols = [col for col in ("lat_gsm", "lon_gsm") if col in df.columns]
                        if drop_cols:
                            df = df.drop(columns=drop_cols)
                        df = df.rename(
                            columns={
                                "bx_gsm": "bx",
                                "by_gsm": "by",
                                "bz_gsm": "bz",
                            }
                        )
                        ordered = [col for col in ("time_tag", "bt", "bx", "by", "bz") if col in df.columns]
                        remaining = [col for col in df.columns if col not in ordered]
                        df = df[ordered + remaining]
                    if target_table == "xray_flux":
                        rename_map = {
                            "irradiance_xrsa": "irradiance_xrsa1",
                            "irradiance_xrsb": "irradiance_xrsb1",
                        }
                        df = df.rename(columns=rename_map)
                        if "irradiance_xrsa1" in df.columns:
                            df["irradiance_xrsa2"] = df["irradiance_xrsa1"]
                        if "irradiance_xrsb1" in df.columns:
                            df["irradiance_xrsb2"] = df["irradiance_xrsb1"]
                        if "xrs_ratio" in df.columns:
                            cols = [col for col in df.columns if col != "xrs_ratio"]
                            df = df[cols + ["xrs_ratio"]]
                    out_conn.execute(f"DROP TABLE IF EXISTS {target_table}")
                    df.to_sql(target_table, out_conn, if_exists="replace", index=False)
    print(f"[OK] Combined {len(db_files)} files into {COMBINED_DB}")


def _delete_db_files(db_files: list[Path]) -> None:
    for path in db_files:
        path.unlink(missing_ok=True)


def main() -> None:
    failures: list[Path] = []
    with ThreadPoolExecutor(max_workers=min(8, len(SCRIPTS))) as executor:
        futures = {executor.submit(_run_script, script): script for script in SCRIPTS}
        for future in as_completed(futures):
            script = futures[future]
            try:
                future.result()
            except Exception:
                failures.append(script)

    db_files = _collect_db_files()
    if not db_files:
        raise RuntimeError("No per-source .db files found to merge.")
    _merge_db_files(db_files)
    _delete_db_files(db_files)
    print("[OK] Removed individual .db files")
    if failures:
        failed_list = ", ".join(str(p.name) for p in failures)
        print(f"[WARN] Failed fetch scripts: {failed_list}")


if __name__ == "__main__":
    main()

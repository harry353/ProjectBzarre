from __future__ import annotations

import argparse
import shutil
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PREPROCESSING_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = PREPROCESSING_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing.utils import (  # noqa: E402
    TARGET_CADENCE,
    engineer_hourly_event_features,
    load_event_tables,
    read_timeseries_table,
    resample_to_hourly,
    write_sqlite_table,
)

# Keep hourly intermediates inside the averaging subdirectory.
HOURLY_OUTPUT_DIR = SCRIPT_DIR / "hourly_outputs"
HOURLY_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Final averaged dataset is stored within this averaging directory.
FINAL_DB = SCRIPT_DIR / "space_weather_aver.db"
START_TIMESTAMP = pd.Timestamp(2005, 1, 1, tz="UTC")

LEGACY_UNIFIED_DB = PREPROCESSING_DIR / "space_weather_merged_aver.db"
LEGACY_TABLE_NAME = "unified_hourly"


@dataclass(frozen=True)
class HourlyDataset:
    name: str
    table_name: str
    output_db: Path
    builder: Callable[["HourlyDataset"], bool]


@dataclass(frozen=True)
class ContinuousConfig:
    table: str
    time_col: str
    value_cols: List[str]
    prefix: str
    resample_method: str


BAD_FILL_THRESHOLD = 1e10
SOLAR_WIND_COLUMNS = ["density", "speed", "temperature"]
IMF_COMPONENTS = {
    "bx": ("bx_gse", "bx"),
    "by": ("by_gse", "by"),
    "bz": ("bz_gse", "bz"),
    "bt": ("bt", "bt"),
}


HOURLY_DATASETS: List[HourlyDataset] = [
    HourlyDataset(
        name="ae",
        table_name="ae_hourly",
        output_db=HOURLY_OUTPUT_DIR / "ae_hourly.db",
        builder=lambda dataset: _build_standard_timeseries(
            dataset,
            table="ae_indices",
            time_col="time_tag",
            value_cols=["al", "au", "ae", "ao"],
            rename_prefix="ae_",
            method="mean",
        ),
    ),
    HourlyDataset(
        name="imf",
        table_name="imf_hourly",
        output_db=HOURLY_OUTPUT_DIR / "imf_hourly.db",
        builder=lambda dataset: _build_imf_hourly(dataset),
    ),
    HourlyDataset(
        name="kp_index",
        table_name="kp_index_hourly",
        output_db=HOURLY_OUTPUT_DIR / "kp_index_hourly.db",
        builder=lambda dataset: _build_standard_timeseries(
            dataset,
            table="kp_index",
            time_col="time_tag",
            value_cols=["kp_index"],
            method="ffill",
        ),
    ),
    HourlyDataset(
        name="dst",
        table_name="dst_hourly",
        output_db=HOURLY_OUTPUT_DIR / "dst_hourly.db",
        builder=lambda dataset: _build_standard_timeseries(
            dataset,
            table="dst_index",
            time_col="time_tag",
            value_cols=["dst"],
            method="ffill",
        ),
    ),
    HourlyDataset(
        name="radio_flux",
        table_name="radio_flux_hourly",
        output_db=HOURLY_OUTPUT_DIR / "radio_flux_hourly.db",
        builder=lambda dataset: _build_standard_timeseries(
            dataset,
            table="radio_flux",
            time_col="time_tag",
            value_cols=["observed_flux", "adjusted_flux", "ursi_flux"],
            rename_prefix="radio_",
            method="ffill",
        ),
    ),
    HourlyDataset(
        name="solar_wind",
        table_name="solar_wind_hourly",
        output_db=HOURLY_OUTPUT_DIR / "solar_wind_hourly.db",
        builder=lambda dataset: _build_solar_wind_hourly(dataset),
    ),
    HourlyDataset(
        name="sunspot_number",
        table_name="sunspot_hourly",
        output_db=HOURLY_OUTPUT_DIR / "sunspot_hourly.db",
        builder=lambda dataset: _build_standard_timeseries(
            dataset,
            table="sunspot_numbers",
            time_col="time_tag",
            value_cols=["sunspot_number"],
            rename_prefix="sunspot_",
            method="ffill",
            extend_final_day=True,
        ),
    ),
    HourlyDataset(
        name="supermag",
        table_name="supermag_hourly",
        output_db=HOURLY_OUTPUT_DIR / "supermag_hourly.db",
        builder=lambda dataset: _build_standard_timeseries(
            dataset,
            table="supermag_indices",
            time_col="time_tag",
            value_cols=["sml", "smu", "sme", "smo"],
            rename_prefix="supermag_",
            method="mean",
        ),
    ),
    HourlyDataset(
        name="sw_comp",
        table_name="sw_comp_hourly",
        output_db=HOURLY_OUTPUT_DIR / "sw_comp_hourly.db",
        builder=lambda dataset: _build_sw_comp_hourly(dataset),
    ),
    HourlyDataset(
        name="xray_flux",
        table_name="xray_flux_hourly",
        output_db=HOURLY_OUTPUT_DIR / "xray_flux_hourly.db",
        builder=lambda dataset: _build_standard_timeseries(
            dataset,
            table="xray_flux",
            time_col="time_tag",
            value_cols=[
                "irradiance_xrsa1",
                "irradiance_xrsa2",
                "irradiance_xrsb1",
                "irradiance_xrsb2",
                "xrs_ratio",
            ],
            rename_prefix="xrs_",
            method="mean",
        ),
    ),
]


CONTINUOUS_CONFIGS: List[ContinuousConfig] = [
    ContinuousConfig("ace_swepam", "time_tag", ["density", "speed", "temperature"], "ace_sw_", "mean"),
    ContinuousConfig("dscovr_f1m", "time_tag", ["density", "speed", "temperature"], "dscovr_sw_", "mean"),
    ContinuousConfig("ace_mfi", "time_tag", ["bx_gse", "by_gse", "bz_gse", "bt"], "ace_imf_", "mean"),
    ContinuousConfig("dscovr_m1m", "time_tag", ["bt", "bx", "by", "bz"], "dscovr_imf_", "mean"),
    ContinuousConfig(
        "xray_flux",
        "time_tag",
        ["irradiance_xrsa1", "irradiance_xrsa2", "irradiance_xrsb1", "irradiance_xrsb2", "xrs_ratio"],
        "xrs_",
        "mean",
    ),
    ContinuousConfig("radio_flux", "time_tag", ["observed_flux", "adjusted_flux", "ursi_flux"], "radio_", "ffill"),
    ContinuousConfig("ae_indices", "time_tag", ["al", "au", "ae", "ao"], "ae_", "mean"),
    ContinuousConfig("dst_index", "time_tag", ["dst"], "dst_", "ffill"),
    ContinuousConfig("kp_index", "time_tag", ["kp_index"], "kp_", "ffill"),
    ContinuousConfig("supermag_indices", "time_tag", ["sml", "smu", "sme", "smo"], "supermag_", "mean"),
    ContinuousConfig("ace_swics_composition", "time_tag", ["o7_o6", "c6_c5", "avg_fe_charge", "fe_to_o"], "sw_comp_", "mean"),
    ContinuousConfig("sunspot_numbers", "time_tag", ["sunspot_number"], "sunspot_", "ffill"),
]


# ---------------------------------------------------------------------------
# Hourly dataset builders


def _build_standard_timeseries(
    dataset: HourlyDataset,
    *,
    table: str,
    time_col: str,
    value_cols: Sequence[str],
    rename_prefix: str | None = None,
    method: str = "mean",
    extend_final_day: bool = False,
) -> bool:
    df = read_timeseries_table(
        table,
        time_col=time_col,
        value_cols=list(value_cols),
        rename_prefix=rename_prefix,
    )
    if df.empty:
        print(f"[WARN] Table '{table}' has no data.")
        return False

    hourly = resample_to_hourly(df, method=method)
    if extend_final_day:
        hourly = _extend_to_final_day(hourly)
    write_sqlite_table(hourly, dataset.output_db, dataset.table_name)
    return True


def _build_solar_wind_hourly(dataset: HourlyDataset) -> bool:
    ace_df = read_timeseries_table(
        "ace_swepam",
        time_col="time_tag",
        value_cols=SOLAR_WIND_COLUMNS,
        rename_prefix="ace_",
    )
    dscovr_df = read_timeseries_table(
        "dscovr_f1m",
        time_col="time_tag",
        value_cols=SOLAR_WIND_COLUMNS,
        rename_prefix="dscovr_",
    )

    if ace_df.empty and dscovr_df.empty:
        print("[WARN] No solar wind datasets were written.")
        return False

    ace_hourly = resample_to_hourly(ace_df, method="mean") if not ace_df.empty else pd.DataFrame()
    dscovr_hourly = (
        resample_to_hourly(dscovr_df, method="mean") if not dscovr_df.empty else pd.DataFrame()
    )

    combined = _combine_time_series(
        [
            ("ace_", ace_hourly),
            ("dscovr_", dscovr_hourly),
        ],
        SOLAR_WIND_COLUMNS,
    )
    if combined.empty:
        print("[WARN] Combined solar wind dataset is empty.")
        return False

    write_sqlite_table(combined, dataset.output_db, dataset.table_name)
    return True


def _build_imf_hourly(dataset: HourlyDataset) -> bool:
    ace_df = read_timeseries_table(
        "ace_mfi",
        time_col="time_tag",
        value_cols=list({name for name, _ in IMF_COMPONENTS.values()}),
        rename_prefix="ace_imf_",
    )
    dscovr_df = read_timeseries_table(
        "dscovr_m1m",
        time_col="time_tag",
        value_cols=list({name for _, name in IMF_COMPONENTS.values()}),
        rename_prefix="dscovr_imf_",
    )

    ace_df = _cleanup_bad_values(ace_df)
    dscovr_df = _cleanup_bad_values(dscovr_df)

    if ace_df.empty and dscovr_df.empty:
        print("[WARN] No IMF datasets were written.")
        return False

    ace_hourly = resample_to_hourly(ace_df, method="mean") if not ace_df.empty else pd.DataFrame()
    dscovr_hourly = (
        resample_to_hourly(dscovr_df, method="mean") if not dscovr_df.empty else pd.DataFrame()
    )

    combined = _combine_imf_components(ace_hourly, dscovr_hourly)
    if combined.empty:
        print("[WARN] Combined IMF dataset is empty.")
        return False

    write_sqlite_table(combined, dataset.output_db, dataset.table_name)
    return True


def _build_sw_comp_hourly(dataset: HourlyDataset) -> bool:
    df = read_timeseries_table(
        "ace_swics_composition",
        time_col="time_tag",
        value_cols=["o7_o6", "c6_c5", "avg_fe_charge", "fe_to_o"],
    )
    df = _cleanup_bad_values(df)
    if df.empty:
        print("[WARN] SW composition table has no data.")
        return False

    hourly = resample_to_hourly(df, method="mean")
    hourly = _extend_to_final_day(hourly)
    write_sqlite_table(hourly, dataset.output_db, dataset.table_name)
    return True


def _load_columns_from_time(tables: Sequence[str], time_col: str) -> pd.DataFrame:
    frames = []
    with sqlite3.connect(DB_PATH) as conn:
        for table in tables:
            cols = conn.execute(f"PRAGMA table_info({table})").fetchall()
            column_names = [row[1] for row in cols]
            if time_col not in column_names:
                print(f"[WARN] Table '{table}' missing time column '{time_col}'.")
                continue
            start_idx = column_names.index(time_col)
            selected = column_names[start_idx:]
            query = f"SELECT {', '.join(selected)} FROM {table}"
            df = pd.read_sql_query(query, conn, parse_dates=[time_col])
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    return combined


# ---------------------------------------------------------------------------
# Master dataset construction (formerly run_master_preprocessing)


def run_all_hourly(selected: Iterable[str] | None = None) -> List[Path]:
    names = {name for name in selected} if selected else None
    produced: List[Path] = []
    for dataset in HOURLY_DATASETS:
        if names and dataset.name not in names:
            continue
        print(f"[INFO] Building hourly dataset: {dataset.name}")
        try:
            success = dataset.builder(dataset)
        except Exception as exc:  # pragma: no cover - raise for visibility
            print(f"[ERROR] Failed to build {dataset.name}: {exc}")
            raise
        if success and dataset.output_db.exists():
            produced.append(dataset.output_db)
        else:
            print(f"[WARN] Hourly DB missing for {dataset.name}: {dataset.output_db}")
    return produced


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

    master = pd.concat(frames, axis=1, join="outer").sort_index()
    master = master[master.notna().any(axis=1)]
    if START_TIMESTAMP is not None:
        master = master.loc[master.index >= START_TIMESTAMP]
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


def _remove_hourly_output_dir() -> None:
    if HOURLY_OUTPUT_DIR.exists():
        shutil.rmtree(HOURLY_OUTPUT_DIR)
        print(f"[INFO] Removed hourly output directory {HOURLY_OUTPUT_DIR}")


# ---------------------------------------------------------------------------
# Legacy "unify_hourly" functionality


def read_continuous_dataset(cfg: ContinuousConfig) -> pd.DataFrame | None:
    df = read_timeseries_table(
        cfg.table,
        time_col=cfg.time_col,
        value_cols=cfg.value_cols,
        rename_prefix=cfg.prefix,
    )
    if df.empty:
        return None
    return resample_to_hourly(df, method=cfg.resample_method)


def build_hourly_index(
    datasets: List[pd.DataFrame],
    event_sources: List[tuple[pd.DataFrame, str]],
) -> pd.DatetimeIndex:
    timestamps = []
    for df in datasets:
        if df is None or df.empty:
            continue
        if isinstance(df.index, pd.DatetimeIndex):
            timestamps.extend([df.index.min(), df.index.max()])
    for df, time_col in event_sources:
        if df is None or df.empty or time_col not in df.columns:
            continue
        timestamps.extend([df[time_col].min(), df[time_col].max()])
    if not timestamps:
        raise ValueError("No timestamps found in provided datasets.")
    start = min(timestamps)
    end = max(timestamps)
    return pd.date_range(start=start.floor(TARGET_CADENCE), end=end.ceil(TARGET_CADENCE), freq=TARGET_CADENCE, tz="UTC")


def merge_all(hourly_index: pd.DatetimeIndex, frames: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    combined = pd.DataFrame(index=hourly_index)
    for df in frames.values():
        if df is None or df.empty:
            continue
        combined = combined.join(df, how="left")
    return combined


def combine_complementary_sources(df: pd.DataFrame) -> Dict[str, List[str]]:
    combinations = {
        "solar_wind_params": [
            ("density", "ace_sw_density", "dscovr_sw_density"),
            ("speed", "ace_sw_speed", "dscovr_sw_speed"),
            ("temperature", "ace_sw_temperature", "dscovr_sw_temperature"),
        ],
        "imf": [
            ("bx", "ace_imf_bx_gse", "dscovr_imf_bx"),
            ("by", "ace_imf_by_gse", "dscovr_imf_by"),
            ("bz", "ace_imf_bz_gse", "dscovr_imf_bz"),
            ("bt", "ace_imf_bt", "dscovr_imf_bt"),
        ],
    }

    created: Dict[str, List[str]] = {}

    for group, specs in combinations.items():
        for suffix, ace_col, dscovr_col in specs:
            ace_series = df.get(ace_col)
            dscovr_series = df.get(dscovr_col)
            if ace_series is None and dscovr_series is None:
                continue
            if ace_series is None:
                combined_series = dscovr_series.copy()
            elif dscovr_series is None:
                combined_series = ace_series.copy()
            else:
                combined_series = ace_series.combine_first(dscovr_series)

            target_col = f"{group}_{suffix}"
            df[target_col] = combined_series
            for original in (ace_col, dscovr_col):
                if original in df.columns:
                    df.drop(columns=original, inplace=True)
            created.setdefault(group, []).append(target_col)
    return created


def save_legacy_output(df: pd.DataFrame) -> None:
    LEGACY_UNIFIED_DB.parent.mkdir(parents=True, exist_ok=True)
    if LEGACY_UNIFIED_DB.exists():
        LEGACY_UNIFIED_DB.unlink()
    payload = df.copy()
    payload.index = payload.index.tz_convert("UTC")
    payload = payload.reset_index().rename(columns={"index": "timestamp"})
    with sqlite3.connect(LEGACY_UNIFIED_DB) as conn:
        payload.to_sql(LEGACY_TABLE_NAME, conn, if_exists="replace", index=False)
    print(f"[INFO] Legacy unified dataset written to {LEGACY_UNIFIED_DB}")


def run_legacy_unify() -> None:
    continuous_frames: Dict[str, pd.DataFrame] = {}
    for cfg in CONTINUOUS_CONFIGS:
        df = read_continuous_dataset(cfg)
        if df is None or df.empty:
            continue
        continuous_frames[cfg.table] = df

    flare_events = load_event_tables(
        ["goes_flares", "goes_flares_archive"],
        time_col="event_time",
        value_cols=["peak_flux_wm2", "background_flux", "integrated_flux"],
        extra_cols=["status", "satellite", "flare_class", "xrsb_flux"],
    )
    cme_events = load_event_tables(
        ["lasco_cme_catalog"],
        time_col="time_tag",
        value_cols=["median_velocity", "angular_width", "min_velocity", "max_velocity", "halo_class"],
    )

    if not continuous_frames and flare_events.empty and cme_events.empty:
        print("[WARN] No datasets available for legacy unification.")
        return

    hourly_index = build_hourly_index(
        list(continuous_frames.values()),
        [(flare_events, "event_time"), (cme_events, "time_tag")],
    )

    flare_features = engineer_hourly_event_features(
        flare_events,
        time_col="event_time",
        value_cols=["peak_flux_wm2", "background_flux", "integrated_flux"],
        prefix="flare_",
        status_col="status",
        prefer_status="event_peak",
        hourly_index=hourly_index,
    )
    cme_features = engineer_hourly_event_features(
        cme_events,
        time_col="time_tag",
        value_cols=["median_velocity", "angular_width", "min_velocity", "max_velocity", "halo_class"],
        prefix="cme_",
        hourly_index=hourly_index,
    )

    all_frames = {**continuous_frames, "flare_features": flare_features, "cme_features": cme_features}
    unified = merge_all(hourly_index, all_frames)
    combined_sources = combine_complementary_sources(unified)
    save_legacy_output(unified)

    summary = {
        "time_range": (unified.index.min(), unified.index.max()),
        "total_columns": len(unified.columns),
    }
    sources = {
        name: list(df.columns)
        for name, df in all_frames.items()
        if df is not None and not df.empty
    }
    for name, cols in combined_sources.items():
        sources[f"{name} (combined)"] = cols
    summary["sources"] = sources
    _print_summary(summary)


def _print_summary(summary: dict) -> None:
    time_start, time_end = summary["time_range"]
    print("[OK] Legacy unified dataset created.")
    print(f"    Time range: {time_start} -> {time_end}")
    print(f"    Total columns: {summary['total_columns']}")
    for name, columns in summary["sources"].items():
        print(f"    {name}: wrote {len(columns)} columns")


# ---------------------------------------------------------------------------
# Shared helpers


def _cleanup_bad_values(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    return df.mask((df > BAD_FILL_THRESHOLD) | (df < -BAD_FILL_THRESHOLD), np.nan)


def _extend_to_final_day(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    last = df.index.max()
    if last is None:
        return df
    if last.tzinfo is None:
        last = last.tz_localize("UTC")
    else:
        last = last.tz_convert("UTC")
    day_end = last.floor("D") + pd.Timedelta(days=1) - pd.Timedelta(hours=1)
    if last >= day_end:
        return df
    full_index = pd.date_range(
        start=df.index.min(),
        end=day_end,
        freq="h",
        tz=df.index.tz or "UTC",
    )
    return df.reindex(full_index).ffill()


def _end_of_final_day(series: pd.Series) -> pd.Timestamp | None:
    if series.empty:
        return None
    last = series.max()
    if pd.isna(last):
        return None
    if last.tzinfo is None:
        last = last.tz_localize("UTC")
    else:
        last = last.tz_convert("UTC")
    return last.floor("D") + pd.Timedelta(days=1) - pd.Timedelta(hours=1)


def _combine_time_series(
    sources: List[tuple[str, pd.DataFrame]],
    columns: Iterable[str],
) -> pd.DataFrame:
    frames = [df for _, df in sources if not df.empty]
    if not frames:
        return pd.DataFrame()

    start = min(df.index.min() for df in frames)
    end = max(df.index.max() for df in frames)
    hourly_index = pd.date_range(
        start=start.floor(TARGET_CADENCE),
        end=end.ceil(TARGET_CADENCE),
        freq=TARGET_CADENCE,
        tz="UTC",
    )
    combined = pd.DataFrame(index=hourly_index)

    for column in columns:
        merged = None
        for prefix, df in sources:
            series = df.get(f"{prefix}{column}")
            if series is None:
                continue
            series = series.reindex(hourly_index)
            if merged is None:
                merged = series
            else:
                merged = merged.combine_first(series)
        combined[column] = merged

    return combined.dropna(how="all")


def _combine_imf_components(ace: pd.DataFrame, dscovr: pd.DataFrame) -> pd.DataFrame:
    frames = [df for df in (ace, dscovr) if not df.empty]
    if not frames:
        return pd.DataFrame()

    start = min(df.index.min() for df in frames)
    end = max(df.index.max() for df in frames)
    hourly_index = pd.date_range(
        start=start.floor(TARGET_CADENCE),
        end=end.ceil(TARGET_CADENCE),
        freq=TARGET_CADENCE,
        tz="UTC",
    )
    combined = pd.DataFrame(index=hourly_index)

    for output_col, (ace_col_raw, dscovr_col_raw) in IMF_COMPONENTS.items():
        ace_series = ace.get(f"ace_imf_{ace_col_raw}")
        dscovr_series = dscovr.get(f"dscovr_imf_{dscovr_col_raw}")
        merged = None
        if ace_series is not None:
            merged = ace_series.reindex(hourly_index)
        if dscovr_series is not None:
            merged = (
                dscovr_series.reindex(hourly_index)
                if merged is None
                else merged.combine_first(dscovr_series.reindex(hourly_index))
            )
        combined[output_col] = merged

    return combined.dropna(how="all")


# ---------------------------------------------------------------------------
# CLI


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build hourly datasets, master dataset, and optional legacy unified datasets.",
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        help="Optional subset of source names to process.",
    )
    parser.add_argument(
        "--hourly-only",
        action="store_true",
        help="Only build per-source hourly databases without assembling the master dataset.",
    )
    parser.add_argument(
        "--skip-cleanup",
        action="store_true",
        help="Keep hourly databases after building the master dataset.",
    )
    parser.add_argument(
        "--legacy-unify",
        action="store_true",
        help="Also build the legacy unified dataset (old unify_hourly output).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected = args.sources
    produced = run_all_hourly(selected)
    if not args.hourly_only:
        if not produced:
            print("[WARN] No hourly databases were produced. Aborting master preprocessing.")
        else:
            master_df = build_master_dataframe(produced)
            if master_df.empty:
                print("[WARN] Master dataframe is empty. Aborting master preprocessing.")
            else:
                write_master_outputs(master_df)
    else:
        print("[INFO] Hourly-only run requested; skipping master dataset assembly.")

    if not args.skip_cleanup:
        cleanup(produced)
        _remove_hourly_output_dir()
    else:
        print("[INFO] Skipping cleanup; hourly databases retained in", HOURLY_OUTPUT_DIR)

    if args.legacy_unify:
        run_legacy_unify()


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd

ESSENTIAL_SOLAR_WIND = ["bz", "by", "speed", "bt"]
SECONDARY_SOLAR_WIND = ["density", "temperature", "o7_o6", "c6_c5", "avg_fe_charge", "fe_to_o"]
GEOMAGNETIC_INDICES = [
    "dst",
    "ae_ae",
    "ae_al",
    "ae_au",
    "supermag_sme",
    "supermag_smu",
    "supermag_sml",
    "supermag_smr",
]
CONTEXTUAL_FEATURES = [
    "radio_observed_flux",
    "radio_adjusted_flux",
    "radio_ursi_flux",
    "sunspot_sunspot_number",
    "f10_7",
    "cme_events_per_hour",
    "cme_event_count_past_24h",
    "cme_event_count_past_48h",
    "cme_time_since_last_event",
    "flare_events_per_hour",
    "flare_event_count_past_24h",
    "flare_event_count_past_48h",
    "flare_time_since_last_event",
]


def impute_space_weather(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply hierarchical, leakage-safe imputation to the provided space-weather dataframe.
    """
    working = df.copy()
    if not isinstance(working.index, pd.DatetimeIndex):
        working.index = pd.to_datetime(working.index, utc=True, errors="coerce")
    working = working.sort_index()
    working = working[~working.index.duplicated(keep="first")]

    dropped_rows, working = _drop_rows_with_geomagnetic_gaps(working)
    essential_masks, interpolated_counts = _impute_essential_solar_wind(working, ESSENTIAL_SOLAR_WIND)
    secondary_masks = _impute_secondary_solar_wind(working, SECONDARY_SOLAR_WIND)
    _zero_fill_contextual_features(working, CONTEXTUAL_FEATURES)
    _handle_engineered_features(working)

    working.attrs["interpolated_counts"] = interpolated_counts
    working.attrs["dropped_rows"] = dropped_rows
    return working


def _load_source_dataframe(db_path: Path) -> pd.DataFrame:
    if not db_path.exists():
        raise FileNotFoundError(f"Source database not found: {db_path}")
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query("SELECT * FROM space_weather_preprocessed", conn, parse_dates=["timestamp"])
    df = df.dropna(subset=["timestamp"])
    df = df.set_index("timestamp").sort_index()
    return df


def _write_target_dataframe(df: pd.DataFrame, db_path: Path) -> None:
    payload = df.reset_index().rename(columns={"index": "timestamp"})
    with sqlite3.connect(db_path) as conn:
        payload.to_sql("space_weather_preprocessed_imp", conn, if_exists="replace", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Hierarchical, leakage-safe imputation for space weather data.")
    parser.add_argument("--source", type=Path, default=SOURCE_DB, help="Path to source averaged DB.")
    parser.add_argument("--target", type=Path, default=TARGET_DB, help="Path for imputed DB.")
    args = parser.parse_args()

    df = _load_source_dataframe(args.source)
    imputed = impute_space_weather(df)
    _write_target_dataframe(imputed, args.target)
    print(f"[OK] Wrote imputed dataset to {args.target}")


def _drop_rows_with_geomagnetic_gaps(df: pd.DataFrame) -> tuple[int, pd.DataFrame]:
    available = [col for col in GEOMAGNETIC_INDICES if col in df.columns]
    if not available:
        return 0, df
    missing_mask = df[available].isna().any(axis=1)
    if missing_mask.any():
        dropped = int(missing_mask.sum())
        df = df.loc[~missing_mask].copy()
        return dropped, df
    return 0, df


def _impute_essential_solar_wind(df: pd.DataFrame, columns: Iterable[str]) -> tuple[Dict[str, pd.Series], Dict[str, int]]:
    masks: Dict[str, pd.Series] = {}
    interpolated_counts: Dict[str, int] = {}
    available = [col for col in columns if col in df.columns]
    if not available:
        return masks, interpolated_counts
    original = df[available].copy()
    for col in columns:
        if col not in df.columns:
            continue
        series = df[col].astype(float)
        interpolated = series.interpolate(method="time", limit=3, limit_direction="both")
        for idx_range, length in _na_runs(series):
            if length <= 3:
                continue
            interpolated.loc[idx_range] = series.loc[idx_range]
        mask = pd.Series(0, index=df.index, dtype=int)
        for idx_range, length in _na_runs(interpolated):
            if 3 < length <= 12:
                mask.loc[idx_range] = 1
                interpolated.loc[idx_range] = 0.0
            elif length > 12:
                mask.loc[idx_range] = 1
                interpolated.loc[idx_range] = 0.0
        df[col] = interpolated
        df[f"{col}_missing_flag"] = mask
        interpolated_counts[col] = int(mask.sum())
        masks[f"{col}_missing_flag"] = mask

    drop_mask = _rows_with_all_missing(original, min_length=13)
    if drop_mask is not None and len(drop_mask) > 0:
        df.drop(index=drop_mask, inplace=True, errors="ignore")
    return masks, interpolated_counts


def _impute_secondary_solar_wind(df: pd.DataFrame, columns: Iterable[str]) -> Dict[str, pd.Series]:
    available = [col for col in columns if col in df.columns]
    if not available:
        zero_mask = pd.Series(0, index=df.index, dtype=int)
        df["sw_secondary_missing_flag"] = zero_mask
        return {"sw_secondary_missing_flag": zero_mask}
    aggregate_mask = pd.Series(0, index=df.index, dtype=int)
    for col in available:
        series = df[col].astype(float)
        interpolated = series.interpolate(method="time", limit=3, limit_direction="both")
        mask = pd.Series(0, index=df.index, dtype=int)
        for idx_range, length in _na_runs(series):
            if length <= 3:
                continue
            mask.loc[idx_range] = 1
            interpolated.loc[idx_range] = 0.0
        df[col] = interpolated.fillna(0.0)
        aggregate_mask = aggregate_mask.reindex(df.index, fill_value=0) | mask.reindex(df.index, fill_value=0)
    aggregate_mask = aggregate_mask.astype(int)
    df["sw_secondary_missing_flag"] = aggregate_mask
    return {"sw_secondary_missing_flag": aggregate_mask}


def _handle_engineered_features(
    df: pd.DataFrame,
) -> None:
    known_columns = set(ESSENTIAL_SOLAR_WIND + SECONDARY_SOLAR_WIND + GEOMAGNETIC_INDICES + CONTEXTUAL_FEATURES)
    engineered_cols = [col for col in df.columns if col not in known_columns]
    for col in engineered_cols:
        df[col] = df[col].fillna(0.0)


def _zero_fill_contextual_features(df: pd.DataFrame, columns: Iterable[str]) -> None:
    for col in columns:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)


def _na_runs(series: pd.Series) -> List[tuple[pd.Index, int]]:
    if series.empty:
        return []
    is_na = series.isna()
    groups = (is_na != is_na.shift()).cumsum()
    runs: List[tuple[pd.Index, int]] = []
    for _, block in is_na.groupby(groups):
        if block.iloc[0]:
            runs.append((block.index, len(block)))
    return runs


def _rows_with_all_missing(original: pd.DataFrame, min_length: int) -> pd.Index | None:
    all_missing = original.isna().all(axis=1)
    if not all_missing.any():
        return None
    groups = (all_missing != all_missing.shift()).cumsum()
    drop_indices: List[pd.Index] = []
    for _, block in all_missing.groupby(groups):
        if block.iloc[0] and len(block) >= min_length:
            drop_indices.append(block.index)
    if not drop_indices:
        return None
    combined = drop_indices[0]
    for idx in drop_indices[1:]:
        combined = combined.append(idx)
    return combined


PIPELINE_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DB = PIPELINE_ROOT / "1_averaging" / "space_weather_aver.db"
TARGET_DB = Path(__file__).resolve().parent / "space_weather_aver_imp.db"


def _load_source_dataframe(db_path: Path) -> pd.DataFrame:
    if not db_path.exists():
        raise FileNotFoundError(f"Source database not found: {db_path}")
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query("SELECT * FROM space_weather_preprocessed", conn, parse_dates=["timestamp"])
    df = df.dropna(subset=["timestamp"])
    df = df.set_index("timestamp").sort_index()
    return df


def _write_target_dataframe(df: pd.DataFrame, db_path: Path) -> None:
    payload = df.reset_index().rename(columns={"index": "timestamp"})
    with sqlite3.connect(db_path) as conn:
        payload.to_sql("space_weather_preprocessed_imp", conn, if_exists="replace", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Hierarchical, leakage-safe imputation for space weather data.")
    parser.add_argument("--source", type=Path, default=SOURCE_DB, help="Path to source averaged DB.")
    parser.add_argument("--target", type=Path, default=TARGET_DB, help="Path for imputed DB.")
    args = parser.parse_args()

    print(f"[INFO] Loading source dataset from {args.source}")
    df = _load_source_dataframe(args.source)
    print("[INFO] Running hierarchical imputation...")
    imputed = impute_space_weather(df)
    dropped_rows = imputed.attrs.get("dropped_rows", 0)
    interpolated_counts = imputed.attrs.get("interpolated_counts", {})
    print(f"[INFO] Dropped rows due to geomagnetic gaps: {dropped_rows}")
    for var, count in interpolated_counts.items():
        print(f"[INFO] Interpolated gaps for {var}: {count}")
    print(f"[INFO] Writing imputed dataset to {args.target}")
    _write_target_dataframe(imputed, args.target)
    print("[OK] Imputation complete.")


if __name__ == "__main__":
    main()

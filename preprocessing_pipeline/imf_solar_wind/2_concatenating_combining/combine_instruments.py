from __future__ import annotations

import sys
from pathlib import Path

# Resolve absolute path of the current script and search for project root (space_weather_api.py)
PROJECT_ROOT = Path(__file__).resolve()
for parent in PROJECT_ROOT.parents:
    if (parent / "space_weather_api.py").exists():
        PROJECT_ROOT = parent
        break
else:
    PROJECT_ROOT = PROJECT_ROOT.parent

# Inject project root into system path to allow local module imports
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from preprocessing_pipeline.utils import load_hourly_output, write_sqlite_table

STAGE_DIR = Path(__file__).resolve().parent
# Load databases from the averaging stage
AVERAGING_DIR = STAGE_DIR.parents[1] / "imf_solar_wind" / "1_averaging"
OUTPUT_DB = STAGE_DIR / "imf_solar_wind_aver_comb.db"

SW_COLUMNS = ["density", "speed", "temperature"]
IMF_COLUMNS = {
    "ace_mfi_aver.db": ["bx_gse", "by_gse", "bz_gse", "bt"],
    "dscovr_m1m_aver.db": ["bx", "by", "bz", "bt"],
}

# Unique identifiers to track source instruments after merging
SOURCE_CODE = {"ace": 1, "dscovr": 2}


# Standard loader for staging databases
def _load(path: Path, table: str) -> pd.DataFrame:
    df = load_hourly_output(path, table)
    if df.empty:
        raise RuntimeError(f"Required hourly dataset missing: {path}")
    return df


# Merges two instrument datasets, giving precedence to one while tracking sources
def combine_with_priority(
    primary: pd.DataFrame,
    secondary: pd.DataFrame,
    primary_label: str,
    secondary_label: str,
) -> pd.DataFrame:
    # Build a unified timeline across both datasets
    index = primary.index.union(secondary.index).sort_values()
    data_columns = sorted(set(primary.columns).union(secondary.columns))
    primary = primary.reindex(index).reindex(columns=data_columns)
    secondary = secondary.reindex(index).reindex(columns=data_columns)

    # Use primary values where available; fill gaps with secondary values
    combined = primary.combine_first(secondary)
    combined = combined.sort_index()

    # Create source metadata columns for each data parameter
    for column in data_columns:
        source_col = pd.Series(0, index=combined.index, dtype="int64", name=f"{column}_source_id")
        primary_mask = primary[column].notna()
        # Mark timestamps where primary instrument provided the value
        source_col.loc[primary_mask] = SOURCE_CODE[primary_label]
        # Mark timestamps where secondary instrument filled a gap
        secondary_mask = ~primary_mask & secondary[column].notna()
        source_col.loc[secondary_mask] = SOURCE_CODE[secondary_label]
        combined[source_col.name] = source_col.astype("int64")

    return combined


# Merges ACE SWEPAM and DSCOVR SW data with DSCOVR priority
def combine_solar_wind() -> pd.DataFrame:
    ace = _load(AVERAGING_DIR / "ace_swepam_aver.db", "hourly_data")
    dscovr = _load(AVERAGING_DIR / "dscovr_f1m_aver.db", "hourly_data")
    combined = combine_with_priority(dscovr, ace, "dscovr", "ace")
    source_cols = [f"{col}_source_id" for col in SW_COLUMNS]
    combined = combined[SW_COLUMNS + source_cols]
    return combined


# Merges ACE MFI and DSCOVR IMF data with DSCOVR priority and schema mapping
def combine_imf() -> pd.DataFrame:
    ace = _load(AVERAGING_DIR / "ace_mfi_aver.db", "hourly_data")
    dscovr = _load(AVERAGING_DIR / "dscovr_m1m_aver.db", "hourly_data")

    # Align diverse column naming conventions to a shared GSE-like schema
    dscovr = dscovr.rename(columns={"bx": "bx_gse", "by": "by_gse", "bz": "bz_gse"})
    if "bx_gsm" in ace.columns:
        ace = ace.rename(columns={"bx_gsm": "bx_gse", "by_gsm": "by_gse", "bz_gsm": "bz_gse"})
    combined = combine_with_priority(dscovr, ace, "dscovr", "ace")
    columns = ["bx_gse", "by_gse", "bz_gse", "bt"]
    source_cols = [f"{col}_source_id" for col in columns]
    combined = combined[columns + source_cols]
    return combined


def main() -> None:
    # Generate combined SW and IMF datasets independently
    solar = combine_solar_wind()
    imf = combine_imf()
    # Join into a single multi-instrument feature matrix
    combined = solar.join(imf, how="outer").sort_index()
    # Ensure source IDs are strictly integer typed (0 = missing)
    for column in [col for col in combined.columns if col.endswith("_source_id")]:
        combined[column] = combined[column].fillna(0).astype("int64")
    # Persist the final combined telemetry
    write_sqlite_table(combined, OUTPUT_DB, "hourly_data")
    print(f"[OK] Combined IMF + solar wind saved to {OUTPUT_DB}")


if __name__ == "__main__":
    main()

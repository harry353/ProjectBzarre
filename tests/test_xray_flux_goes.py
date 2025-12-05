import os
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

# ------------------------------------------------------------
# Ensure project root is importable
# ------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ------------------------------------------------------------
# Imports
# ------------------------------------------------------------
from data_sources.xray_flux.xray_flux_goes_data_source import XRayFluxGOESDataSource
from data_sources.xray_flux.xray_flux_goes_plot import plot_xrs_goes


# ------------------------------------------------------------
# Test helper
# ------------------------------------------------------------
def run_case(description, days):
    print("\n--------------------------------------------------")
    print(f"TEST: {description}")
    print("--------------------------------------------------")

    ds = XRayFluxGOESDataSource(days=days)
    print(f"Date range: {ds.range_str()}")

    print("Downloading...")
    df = ds.download()
    print(f"Downloaded rows: {len(df)}")

    if df.empty:
        print("No data returned. Skipping ingestion and plotting.")
        return

    print("\n[DEBUG] XRS GOES dataframe dtypes:")
    print(df.dtypes.to_string())
    print("\n[DEBUG] First rows (full width):")
    with pd.option_context(
        "display.max_rows",
        None,
        "display.max_columns",
        None,
        "display.width",
        None,
    ):
        print(df.head())
    print("\nDataFrame schema (column -> dtype):")
    print(df.dtypes.to_string())
    print("\nFirst rows:")
    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", None):
        print(df.head())

    print("Ingesting into test DB...")
    db_path = "test_xray.db"
    if os.path.exists(db_path):
        os.remove(db_path)

    inserted = ds.ingest(df, db_path=db_path)
    print(f"Inserted rows: {inserted}")

    print("Plotting...")
    plot_xrs_goes(df)
    print("Plot complete.")


# ------------------------------------------------------------
# Entry point
# ------------------------------------------------------------

def main():
    run_case("Integer days = 7", 7)

if __name__ == "__main__":
    main()

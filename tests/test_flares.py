import os
import sys
from datetime import date
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_sources.flares.flares_data_source import FlaresDataSource


def run_case(description, days):
    print("\n--------------------------------------------------")
    print(f"FLARES TEST: {description}")
    print("--------------------------------------------------")

    ds = FlaresDataSource(days=days)
    print(f"Date range: {ds.range_str()}")

    print("Downloading...")
    df = ds.download()
    print(f"Downloaded rows: {len(df)}")

    if df.empty:
        print("No data returned. Skipping ingestion and plotting.")
        return

    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        print(df.head())

    print("Ingesting into test DB...")
    db_path = "test_flares.db"
    if os.path.exists(db_path):
        os.remove(db_path)

    inserted = ds.ingest(df, db_path=db_path)
    print(f"Inserted rows: {inserted}")

    print("Plotting...")
    ds.plot(df)
    print("Plot complete.")


def main():
    run_case("GOES flare summary sample", (date(2019, 2, 1), date(2019, 2, 2)))


if __name__ == "__main__":
    main()

from datetime import date
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_sources.sw_comp.sw_comp_data_source import SWCompDataSource


def run_case(description, days):
    print("\n--------------------------------------------------")
    print(f"SW COMP TEST: {description}")
    print("--------------------------------------------------")

    ds = SWCompDataSource(days=days)
    print(f"Date range: {ds.range_str()}")

    print("Downloading...")
    df = ds.download()
    print(f"Downloaded rows: {len(df)}")

    if df.empty:
        print("No data returned. Skipping ingestion and plotting.")
        return

    print("Ingesting into test DB...")
    db_path = "test_sw_comp.db"
    if os.path.exists(db_path):
        os.remove(db_path)

    inserted = ds.ingest(df, db_path=db_path)
    print(f"Inserted rows: {inserted}")

    print("Plotting...")
    ds.plot(df)
    print("Plot complete.")


def main():
    run_case(
        "ACE/SWICS: March 16 2025",
        (date(2025, 3, 15), date(2025, 3, 16)),
    )


if __name__ == "__main__":
    main()

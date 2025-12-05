from datetime import date
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_sources.imf_ace.imf_ace_data_source import IMFACEDataSource


def run_case(description, days):
    print("\n--------------------------------------------------")
    print(f"IMF ACE TEST: {description}")
    print("--------------------------------------------------")

    ds = IMFACEDataSource(days=days)
    print(f"Date range: {ds.range_str()}")

    print("Downloading...")
    df = ds.download()
    print(f"Downloaded rows: {len(df)}")

    if df.empty:
        print("No data returned. Skipping ingestion and plotting.")
        return

    print("Ingesting into test DB...")
    db_path = "test_imf_ace.db"
    if os.path.exists(db_path):
        os.remove(db_path)

    inserted = ds.ingest(df, db_path=db_path)
    print(f"Inserted rows: {inserted}")

    print("Plotting...")
    ds.plot(df)
    print("Plot complete.")


def main():
    run_case(
        "ACE MFI H3 on 2025-10-07",
        days=(date(2025, 11, 1), date(2025, 11, 5)),
    )


if __name__ == "__main__":
    main()

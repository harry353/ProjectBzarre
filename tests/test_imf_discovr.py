from datetime import date
import sys
from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_sources.imf_discovr.imf_discovr_data_source import IMFDiscovrDataSource


def run_case(description, days):
    print("\n--------------------------------------------------")
    print(f"IMF DSCOVR TEST: {description}")
    print("--------------------------------------------------")

    ds = IMFDiscovrDataSource(days=days)
    print(f"Date range: {ds.range_str()}")

    print("Downloading...")
    df = ds.download()
    print(f"Downloaded rows: {len(df)}")

    if df.empty:
        print("No data returned. Skipping ingestion and plotting.")
        return

    print("Ingesting into test DB...")
    db_path = "test_imf_discovr.db"
    if os.path.exists(db_path):
        os.remove(db_path)

    inserted = ds.ingest(df, db_path=db_path)
    print(f"Inserted rows: {inserted}")

    print("Plotting...")
    ds.plot(df)
    print("Plot complete.")


def main():
    run_case("Integer days = 7", (date(2017, 1, 1), date(2017, 1, 7)))


if __name__ == "__main__":
    main()

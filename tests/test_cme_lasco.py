import os
import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_sources.cme_lasco.cme_lasco_data_source import CMELASCODataSource


def run_case(description, days):
    print("\n--------------------------------------------------")
    print(f"CME LASCO TEST: {description}")
    print("--------------------------------------------------")

    ds = CMELASCODataSource(days=days)
    print(f"Date range: {ds.range_str()}")

    print("Downloading...")
    df = ds.download()
    print(f"Downloaded rows: {len(df)}")

    if df.empty:
        print("No data returned. Skipping ingestion and plotting.")
        return

    print("Ingesting into test DB...")
    db_path = "test_cme_lasco.db"
    if os.path.exists(db_path):
        os.remove(db_path)

    inserted = ds.ingest(df, db_path=db_path)
    print(f"Inserted rows: {inserted}")

    print("Plotting...")
    ds.plot(df)
    print("Plot complete.")


def main():
    historic_range = (date(2012, 1, 1), date(2012, 1, 1))
    run_case("Historic Jan 2012 range", historic_range)


if __name__ == "__main__":
    main()

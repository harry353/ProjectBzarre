from datetime import date
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_sources.ae.ae_data_source import AEDataSource


def run_case(description, days):
    print("\n--------------------------------------------------")
    print(f"AE TEST: {description}")
    print("--------------------------------------------------")

    ds = AEDataSource(days=days)
    print(f"Date range: {ds.range_str()}")

    print("Downloading...")
    df = ds.download()
    print(f"Downloaded rows: {len(df)}")

    if df.empty:
        print("No data returned. Skipping ingestion and plotting.")
        return

    print("Ingesting into test DB...")
    db_path = "test_ae.db"
    if os.path.exists(db_path):
        os.remove(db_path)
    inserted = ds.ingest(df, db_path=db_path)
    print(f"Inserted rows: {inserted}")

    print("Plotting...")
    ds.plot(df)
    print("Plot complete.")


def main():
    run_case("Integer days = 1", (date(2020, 12, 28), date(2021, 1, 2)))


if __name__ == "__main__":
    main()


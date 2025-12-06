import os
import sys
from datetime import date
from pathlib import Path
from typing import Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_sources.cme.cme_lasco_data_source import CMELASCODataSource


def run_case(description: str, days: int | Tuple[date, date]):
    print("\n--------------------------------------------------")
    print(f"CME TEST: {description}")
    print("--------------------------------------------------")

    ds = CMELASCODataSource(days=days)
    print(f"Source: {CMELASCODataSource.__name__}")
    print(f"Date range: {ds.range_str()}")

    print("Downloading...")
    df = ds.download()
    print(f"Downloaded rows: {len(df)}")

    if df.empty:
        print("No data returned. Skipping ingestion and plotting.")
        return

    try:
        db_path = "test_cme_lasco.db"
        if os.path.exists(db_path):
            os.remove(db_path)

        inserted = ds.ingest(df, db_path=db_path)
        print(f"Inserted rows: {inserted}")

        print("Plotting...")
        ds.plot(df)
        print("Plot complete.")
    finally:
        if os.path.exists(db_path):
            os.remove(db_path)


def main():
    run_case("LASCO Jan 2000 sample", (date(2025, 12, 3), date(2025, 12, 4)))


if __name__ == "__main__":
    main()

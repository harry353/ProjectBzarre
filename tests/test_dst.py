from datetime import date
import os
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_sources.dst.dst_data_source import DstDataSource


def run_case(description, days):
    print("\n--------------------------------------------------")
    print(f"DST TEST: {description}")
    print("--------------------------------------------------")

    ds = DstDataSource(days=days)
    print(f"Date range: {ds.range_str()}")

    print("Downloading...")
    df = ds.download()

    if not df.empty and "time_tag" in df.columns:
        df = df.copy()
        df["time_tag"] = pd.to_datetime(df["time_tag"], errors="coerce", utc=True)
        df["time_tag"] = df["time_tag"].dt.strftime("%Y-%m-%d %H:%M:%S%z").str.replace(
            "+0000", "+00:00", regex=False
        )
    print(f"Downloaded rows: {len(df)}")
    print(df)

    if df.empty:
        print("No data returned. Skipping ingestion and plotting.")
        return

    print("Ingesting into test DB...")
    db_path = "test_dst.db"
    if os.path.exists(db_path):
        os.remove(db_path)
    inserted = ds.ingest(df, db_path=db_path)
    print(f"Inserted rows: {inserted}")

    print("Plotting...")
    ds.plot(df)
    print("Plot complete.")

def main():
    run_case("Integer days = 7", (date(2026, 1, 15), date(2026, 1, 16)))


if __name__ == "__main__":
    main()

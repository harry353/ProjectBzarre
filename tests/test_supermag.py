from datetime import date
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_sources.supermag.supermag_data_source import SuperMAGDataSource

SUPERMAG_LOGON = "haris262"


def run_case(description, days, logon):
    print("\n--------------------------------------------------")
    print(f"SUPERMAG TEST: {description}")
    print("--------------------------------------------------")

    ds = SuperMAGDataSource(days=days, logon=logon)
    print(f"Date range: {ds.range_str()}")

    print("Downloading...")
    df = ds.download()
    print(f"Downloaded rows: {len(df)}")

    if df.empty:
        print("No data returned. Skipping ingestion and plotting.")
        return

    print("Ingesting into test DB...")
    db_path = "test_supermag.db"
    if os.path.exists(db_path):
        os.remove(db_path)

    inserted = ds.ingest(df, db_path=db_path)
    print(f"Inserted rows: {inserted}")

    print("Plotting...")
    ds.plot(df)
    print("Plot complete.")


def main():
    logon = SUPERMAG_LOGON
    run_case(
        "Explicit date range",
        (date(2024, 11, 20), date(2024, 11, 21)),
        logon,
    )


if __name__ == "__main__":
    main()

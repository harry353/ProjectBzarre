import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_sources.radio_flux.radio_flux_data_source import RadioFluxDataSource


def run_case(description, days):
    print("\n--------------------------------------------------")
    print(f"RADIO FLUX TEST: {description}")
    print("--------------------------------------------------")

    ds = RadioFluxDataSource(days=days)
    print(f"Date range: {ds.range_str()}")

    print("Downloading...")
    df = ds.download()
    print(f"Downloaded rows: {len(df)}")

    if df.empty:
        print("No data returned. Skipping ingestion and plotting.")
        return

    print("Ingesting into test DB...")
    db_path = "test_radio_flux.db"
    if os.path.exists(db_path):
        os.remove(db_path)
    inserted = ds.ingest(df, db_path=db_path)
    print(f"Inserted rows: {inserted}")

    print("Plotting...")
    ds.plot(df)
    print("Plot complete.")


def main():
    run_case("Integer days = 7", 7)


if __name__ == "__main__":
    main()


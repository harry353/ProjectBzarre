import os
import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_sources.solar_wind_ace.solar_wind_ace_data_source import SolarWindACEDataSource


def run_case(description, days):
    print("\n--------------------------------------------------")
    print(f"SOLAR WIND TEST: {description}")
    print("--------------------------------------------------")

    ds = SolarWindACEDataSource(days=days)
    print(f"Date range: {ds.range_str()}")

    print("Downloading...")
    df = ds.download()
    print(f"Downloaded rows: {len(df)}")

    if df.empty:
        print("No data returned. Skipping ingestion and plotting.")
        return

    print("Ingesting into test DB...")
    db_path = "test_solar_wind_ace.db"
    if os.path.exists(db_path):
        os.remove(db_path)
    inserted = ds.ingest(df.copy(), db_path=db_path)
    print(f"Inserted rows: {inserted}")

    print("Plotting...")
    ds.plot(df)
    print("Plot complete.")


def main():
    run_case("ACE SWEPAM sample", (date(1998, 1, 1), date(1998, 3, 4)))


if __name__ == "__main__":
    main()

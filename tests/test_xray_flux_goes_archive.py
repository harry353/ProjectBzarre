import os
import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_sources.xray_flux.xray_flux_goes_archive_data_source import (
    XRayFluxGOESArchiveDataSource,
)
from data_sources.xray_flux.xray_flux_goes_plot import plot_xrs_goes


def run_case(description, days):
    print("\n--------------------------------------------------")
    print(f"ARCHIVE XRS TEST: {description}")
    print("--------------------------------------------------")

    ds = XRayFluxGOESArchiveDataSource(days=days)
    print(f"Date range: {ds.range_str()}")

    print("Downloading...")
    df = ds.download()
    print(f"Downloaded rows: {len(df)}")

    if df.empty:
        print("No archive data returned. Skipping ingestion and plotting.")
        return

    print("Ingesting into test DB..." )
    db_path = "test_xray_archive.db"
    if os.path.exists(db_path):
        os.remove(db_path)

    inserted = ds.ingest(df, db_path=db_path)
    print(f"Inserted rows: {inserted}")

    print("Plotting...")
    plot_xrs_goes(df)
    print("Plot complete.")


def main():
    run_case("GOES-10 example", (date(2005, 1, 1), date(2005, 1, 3)))


if __name__ == "__main__":
    main()

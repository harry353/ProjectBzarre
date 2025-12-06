import os
import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_sources.xray_flux.xray_flux_goes_data_source import XRayFluxGOESDataSource


def run_case(description, days):
    print("\n--------------------------------------------------")
    print(f"XRS TEST: {description}")
    print("--------------------------------------------------")

    ds = XRayFluxGOESDataSource(days=days)
    print(f"Date range: {ds.range_str()}")

    print("Downloading...")
    df = ds.download()
    print(f"Downloaded rows: {len(df)}")

    if df.empty:
        print("No data returned. Skipping ingestion and plotting.")
        return

    datasets = sorted(df["dataset"].dropna().unique()) if "dataset" in df.columns else ["realtime"]
    print(f"Datasets detected: {', '.join(datasets)}")

    for dataset in datasets:
        print(f"Ingesting {dataset} segment...")
        db_path = f"test_xray_flux_{dataset}.db"
        if os.path.exists(db_path):
            os.remove(db_path)
        subset = df if "dataset" not in df.columns else df[df["dataset"] == dataset]
        inserted = ds.ingest(subset.copy(), db_path=db_path)
        print(f"[{dataset}] Inserted rows: {inserted}")

    print("Plotting...")
    ds.plot(df)
    print("Plot complete.")


def main():
    run_case("GOES realtime sample", (date(2024, 2, 6), date(2024, 2, 7)))
    run_case("GOES archive sample", (date(2005, 1, 1), date(2005, 1, 3)))


if __name__ == "__main__":
    main()

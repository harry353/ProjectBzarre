import os
import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_sources.solar_wind.solar_wind_data_source import SolarWindDataSource


def run_case(description, days):
    print("\n--------------------------------------------------")
    print(f"SOLAR WIND TEST: {description}")
    print("--------------------------------------------------")

    ds = SolarWindDataSource(days=days)
    print(f"Date range: {ds.range_str()}")

    print("Downloading...")
    df = ds.download()
    print(f"Downloaded rows: {len(df)}")

    if df.empty:
        print("No data returned. Skipping ingestion and plotting.")
        return

    datasets = (
        sorted(df["dataset"].dropna().unique())
        if "dataset" in df.columns
        else ["dscovr"]
    )
    print(f"Datasets detected: {', '.join(datasets)}")

    print("Ingesting into test DB...")
    for dataset in datasets:
        db_path = f"test_solar_wind_{dataset}.db"
        if os.path.exists(db_path):
            os.remove(db_path)

        subset = df if "dataset" not in df.columns else df[df["dataset"] == dataset]
        inserted = ds.ingest(subset.copy(), db_path=db_path)
        print(f"[{dataset}] Inserted rows: {inserted}")

    print("Plotting...")
    ds.plot(df)
    print("Plot complete.")

    for dataset in datasets:
        db_path = f"test_solar_wind_{dataset}.db"
        if os.path.exists(db_path):
            os.remove(db_path)


def main():
    run_case("ACE SWEPAM sample", (date(2002, 12, 13), date(2002, 12, 14)))
    run_case("DSCOVR sample", (date(2024, 2, 1), date(2024, 2, 3)))


if __name__ == "__main__":
    main()

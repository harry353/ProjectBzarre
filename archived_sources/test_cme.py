import os
import sys
from datetime import date
from pathlib import Path
from typing import Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from archived_sources.cme_donki_lasco.cme_data_source import CMEDataSource


def run_case(description: str, days: int | Tuple[date, date]):
    print("\n--------------------------------------------------")
    print(f"CME TEST: {description}")
    print("--------------------------------------------------")

    ds = CMEDataSource(days=days)
    print(f"Source: {CMEDataSource.__name__}")
    print(f"Date range: {ds.range_str()}")

    print("Downloading...")
    df = ds.download()
    print(f"Downloaded rows: {len(df)}")

    if df.empty:
        print("No data returned. Skipping ingestion and plotting.")
        return

    try:
        datasets = (
            sorted(df["dataset"].dropna().unique())
            if "dataset" in df.columns
            else ["lasco"]
        )
        print(f"Datasets detected: {', '.join(datasets)}")

        print("Ingesting into test DB...")
        for dataset in datasets:
            db_path = f"test_cme_{dataset}.db"
            if os.path.exists(db_path):
                os.remove(db_path)
            subset = df if "dataset" not in df.columns else df[df["dataset"] == dataset]
            inserted = ds.ingest(subset.copy(), db_path=db_path)
            print(f"[{dataset}] Inserted rows: {inserted}")

        print("Plotting...")
        ds.plot(df)
        print("Plot complete.")
    finally:
        for dataset in ["lasco", "donki"]:
            db_path = f"test_cme_{dataset}.db"
            if os.path.exists(db_path):
                os.remove(db_path)


def main():
    run_case("LASCO Jan 2005 sample", (date(2005, 1, 1), date(2005, 1, 3)))
    run_case("DONKI Feb 2024 sample", (date(2024, 2, 1), date(2024, 2, 3)))


if __name__ == "__main__":
    main()

from datetime import date
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_sources.euv_images_soho.euv_data_source import EUVImagesSOHODataSource


def run_case(description: str, days) -> None:
    print("\n==================================================")
    print(f"EUV IMAGES (SOHO) TEST: {description}")
    print("==================================================")

    ds = EUVImagesSOHODataSource(days)
    db_path = "test_euv_images.db"
    if os.path.exists(db_path):
        os.remove(db_path)

    stats = ds.download_ingest_cleanup(batch_size=5, db_path=db_path)
    print(f"Batch processing stats: {stats}")
    if stats["downloaded"] == 0:
        print("No files retrieved; aborting plot.")
        return

    preview_ds = EUVImagesSOHODataSource((ds.start_date, ds.start_date))
    preview_files = preview_ds.download()
    if not preview_files:
        print("No preview files available for plotting.")
        return

    print("Plotting first FITS file...")
    ds.plot(preview_files)
    print("Plot complete.")

    for file in preview_files:
        try:
            file.unlink()
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    run_case("SOHO EIT sample window", (date(2012, 6, 1), date(2012, 6, 1)))

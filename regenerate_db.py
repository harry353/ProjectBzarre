from datetime import date
import time
import importlib
import inspect
import os
import pkgutil
from pathlib import Path

from space_weather_api import SpaceWeatherAPI
from space_weather_warehouse import SpaceWeatherWarehouse

DB_PATH = "space_weather.db"
DATA_SOURCES_DIR = Path("data_sources")
MODULE_SUFFIX = "_data_source"


def reset_database():
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)


def iter_data_source_modules():
    """
    Yield fully qualified module names that end with `_data_source` anywhere
    under the data_sources/ package.
    """
    prefix = "data_sources."
    for module in pkgutil.walk_packages([str(DATA_SOURCES_DIR)], prefix=prefix):
        if module.ispkg:
            continue
        if module.name.split(".")[-1].endswith(MODULE_SUFFIX):
            yield module.name


def load_data_source_classes():
    classes = []
    for module_name in iter_data_source_modules():
        module = importlib.import_module(module_name)
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if issubclass(obj, SpaceWeatherAPI) and obj is not SpaceWeatherAPI:
                classes.append(obj)
    return classes


def process_sources(days):
    warehouse = SpaceWeatherWarehouse(DB_PATH)
    classes = load_data_source_classes()
    success_count = 0
    failure_count = 0
    failed_sources = []

    print(f"Found {len(classes)} data source(s):")
    for cls in classes:
        print(f"  - {cls.__module__}.{cls.__name__}")
    print()

    for cls in classes:
        print(f"Processing {cls.__name__} ({cls.__module__})...")
        try:
            kwargs = {"days": days}
            if cls.__name__ == "SuperMAGDataSource":
                kwargs["logon"] = "haris262"
            source = cls(**kwargs)
            df = source.download()
            inserted = source.ingest(df, warehouse=warehouse)
            print(f"{cls.__name__}: inserted {inserted} rows.")
            if inserted > 0:
                success_count += 1
            else:
                failure_count += 1
                failed_sources.append(cls.__name__)
                print(f"[WARN] {cls.__name__} inserted 0 rows.")
        except Exception as exc:
            print(f"[ERROR] {cls.__name__} failed: {exc}")
            failure_count += 1
            failed_sources.append(cls.__name__)
        print()

    print(
        f"Summary: {success_count} data sources succeeded, {failure_count} failed."
    )
    if failed_sources:
        print("Failed data sources:")
        for name in failed_sources:
            print(f"  - {name}")

def main(days=(date(2000, 1, 1), date(2025, 3, 5))):
    start_time = time.time()
    reset_database()
    process_sources(days)
    duration = time.time() - start_time
    minutes = duration / 60
    num_days = (days[1] - days[0]).days + 1 if isinstance(days, tuple) else None
    per_day = duration / num_days if num_days and num_days > 0 else None
    message = f"Database update complete in {duration:.2f} seconds ({minutes:.2f} minutes)"
    if per_day:
        message += f", averaging {per_day:.2f} seconds per day."
    else:
        message += "."
    print(message)


if __name__ == "__main__":
    main()

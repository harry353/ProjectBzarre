from datetime import date, datetime, timedelta
import importlib
import inspect
import io
import pkgutil
import sys
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from space_weather_api import SpaceWeatherAPI

LOOKBACK_DAYS = 30
TIME_COLUMNS_MAP = {
    "CMEDataSource": ["time21_5", "startTime"],
    "CMELASCODataSource": ["datetime_utc"],
    "SolarFlareDataSource": ["endTime", "peakTime", "beginTime"],
    "XRayFluxGOESDataSource": "__index__",
    "XRayFluxGOESArchiveDataSource": "__index__",
}
DIRECTORY_URLS = {
    "AEDataSource": "https://wdc.kugi.kyoto-u.ac.jp/ae_realtime/data_dir/",
    "CMEDataSource": "https://kauai.ccmc.gsfc.nasa.gov/DONKI/WS/get/CMEAnalysis",
    "CMELASCODataSource": "https://cdaw.gsfc.nasa.gov/CME_list/UNIVERSAL_ver2/text_ver/",
    "DstDataSource": "https://wdc.kugi.kyoto-u.ac.jp/dst_provisional/",
    "IMFACEDataSource": "https://cdaweb.gsfc.nasa.gov/pub/data/ace/mag/level_2_cdaweb/mfi_h3/",
    "IMFDiscovrDataSource": "https://www.ngdc.noaa.gov/dscovr/data/",
    "KpIndexDataSource": "https://kp.gfz.de/app/json/",
    "RadioFluxDataSource": "https://www.spaceweather.gc.ca/solar_flux_data/daily_flux_values/",
    "SolarFlareDataSource": "https://kauai.ccmc.gsfc.nasa.gov/DONKI/WS/get/FLR",
    "SolarWindDataSource": "https://www.ngdc.noaa.gov/dscovr/data/",
    "SunspotNumberDataSource": "https://kp.gfz.de/app/json/",
    "SuperMAGDataSource": "https://supermag.jhuapl.edu/services/indices.php",
    "SWCompDataSource": "https://cdaweb.gsfc.nasa.gov/pub/data/ace/swics/level_2_cdaweb/sw2_h3/",
    "XRayFluxGOESDataSource": "https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/",
    "XRayFluxGOESArchiveDataSource": "https://www.ncei.noaa.gov/data/goes-space-environment-monitor/access/science/xrs/",
}


def iter_data_source_classes():
    package_path = PROJECT_ROOT / "data_sources"
    prefix = "data_sources."
    for module in pkgutil.walk_packages([str(package_path)], prefix=prefix):
        if module.ispkg:
            continue
        if not module.name.split(".")[-1].endswith("_data_source"):
            continue

        mod = importlib.import_module(module.name)
        for _, obj in inspect.getmembers(mod, inspect.isclass):
            if issubclass(obj, SpaceWeatherAPI) and obj is not SpaceWeatherAPI:
                yield obj


def resolve_kwargs(cls):
    today = date.today()
    start = today - timedelta(days=LOOKBACK_DAYS - 1)
    kwargs = {"days": (start, today)}
    if cls.__name__ == "SuperMAGDataSource":
        start = max(today - timedelta(days=6), start)
        kwargs["days"] = (start, today)
        kwargs["logon"] = "haris262"
    return kwargs


def main():
    print("Testing latest release date for all data sources...\n")
    for cls in iter_data_source_classes():
        print(f"-> {cls.__name__}")
        try:
            latest = find_latest_release(cls)
            if latest is None:
                print("   Most recent release: <no data>")
            else:
                directory = DIRECTORY_URLS.get(cls.__name__, "<unknown directory>")
                print(f"   Most recent release: {latest}")
                print(f"   Source directory: {directory}")
        except Exception as exc:
            print(f"   [ERROR] Failed to fetch {cls.__name__}: {exc}")
        print()
    print("\nAll sources processed.")


def determine_latest(class_name, df):
    columns = TIME_COLUMNS_MAP.get(class_name, ["time_tag", "time"])

    if columns == "__index__":
        ts = getattr(df.index, "max", lambda: None)()
        return _to_date(ts)

    if isinstance(columns, str):
        columns = [columns]

    for column in columns:
        if column == "__index__":
            ts = getattr(df.index, "max", lambda: None)()
        else:
            if column not in df.columns:
                continue
            series = df[column].dropna()
            if series.empty:
                continue
            ts = series.max()
        date_value = _to_date(ts)
        if date_value is not None:
            return date_value

    if "time_tag" in df.columns:
        return _to_date(df["time_tag"].max())
    if "time" in df.columns:
        return _to_date(df["time"].max())
    return _to_date(getattr(df.index, "max", lambda: None)())


def _to_date(value):
    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        value = value.to_pydatetime()
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return None


def find_latest_release(cls):
    kwargs = resolve_kwargs(cls)
    ds = cls(**kwargs)
    buffer = io.StringIO()
    with redirect_stdout(buffer), redirect_stderr(buffer):
        df = ds.download()

    if df.empty:
        return None

    return determine_latest(cls.__name__, df)


if __name__ == "__main__":
    main()

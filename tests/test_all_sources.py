from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
import importlib
import inspect
import io
import pkgutil
import re
import sys
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from space_weather_api import SpaceWeatherAPI
from common.http import http_get

LOOKBACK_DAYS = 30
TIME_COLUMNS_MAP = {
    "CMEDataSource": ["time21_5", "startTime"],
    "CMELASCODataSource": ["datetime_utc"],
    "FlaresArchiveDataSource": ["event_time"],
    "FlaresDataSource": ["event_time"],
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
    "FlaresDataSource": "https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/",
    "FlaresArchiveDataSource": "https://www.ncei.noaa.gov/data/goes-space-environment-monitor/access/science/xrs/",
    "SolarFlareDataSource": "https://kauai.ccmc.gsfc.nasa.gov/DONKI/WS/get/FLR",
    "SolarWindDataSource": "https://www.ngdc.noaa.gov/dscovr/data/",
    "SunspotNumberDataSource": "https://kp.gfz.de/app/json/",
    "SuperMAGDataSource": "https://supermag.jhuapl.edu/services/indices.php",
    "SWCompDataSource": "https://cdaweb.gsfc.nasa.gov/pub/data/ace/swics/level_2_cdaweb/sw2_h3/",
    "XRayFluxGOESDataSource": "https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/",
    "XRayFluxGOESArchiveDataSource": "https://www.ncei.noaa.gov/data/goes-space-environment-monitor/access/science/xrs/",
}

DATE_TOKEN_REGEXES = [
    re.compile(r"d(?P<ymd>\d{8})"),
    re.compile(r"(?P<y>\d{4})[-_/](?P<m>\d{2})[-_/](?P<d>\d{2})"),
    re.compile(r"(?<!\d)(?P<ymd>\d{8})(?!\d)"),
]


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
    classes = list(iter_data_source_classes())
    results = _fetch_all_latest_parallel(classes)

    for cls in classes:
        name = cls.__name__
        print(f"-> {name}")
        latest, error = results.get(name, (None, RuntimeError("No result produced")))
        if error is not None:
            print(f"   [ERROR] Failed to fetch {name}: {error}")
        elif latest is None:
            print("   Most recent release: <no data>")
        else:
            directory = DIRECTORY_URLS.get(name, "<unknown directory>")
            print(f"   Most recent release: {latest}")
            print(f"   Source directory: {directory}")
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
    directory = DIRECTORY_URLS.get(cls.__name__)
    if directory:
        latest = _find_latest_from_directory(cls.__name__, directory)
        if latest is not None:
            return latest

    return _find_latest_via_download(cls)


def _fetch_all_latest_parallel(classes):
    if not classes:
        return {}

    # Threaded IO fetch speeds up network bound downloads.
    max_workers = min(8, len(classes)) or 1
    results = {}

    def task(cls):
        try:
            latest = find_latest_release(cls)
            return cls.__name__, latest, None
        except Exception as exc:
            return cls.__name__, None, exc

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_cls = {executor.submit(task, cls): cls for cls in classes}
        for future in as_completed(future_to_cls):
            name, latest, error = future.result()
            results[name] = (latest, error)

    return results


def _find_latest_via_download(cls):
    kwargs = resolve_kwargs(cls)
    ds = cls(**kwargs)
    buffer = io.StringIO()
    with redirect_stdout(buffer), redirect_stderr(buffer):
        df = ds.download()

    if df.empty:
        return None

    return determine_latest(cls.__name__, df)


def _find_latest_from_directory(class_name, directory):
    response = http_get(directory, log_name=class_name, timeout=30)
    if response is None:
        return None

    latest = None
    for candidate in _extract_dates_from_text(response.text):
        if latest is None or candidate > latest:
            latest = candidate
    return latest


def _extract_dates_from_text(text):
    seen = set()
    for pattern in DATE_TOKEN_REGEXES:
        for match in pattern.finditer(text):
            groups = match.groupdict()
            if "ymd" in groups:
                token = groups["ymd"]
                if token in seen:
                    continue
                seen.add(token)
                result = _parse_compact_date(token)
            else:
                token = "-".join([groups["y"], groups["m"], groups["d"]])
                if token in seen:
                    continue
                seen.add(token)
                result = _parse_y_m_d(groups["y"], groups["m"], groups["d"])

            if result is not None:
                yield result


def _parse_compact_date(token):
    try:
        return datetime.strptime(token, "%Y%m%d").date()
    except ValueError:
        return None


def _parse_y_m_d(y, m, d):
    try:
        return date(int(y), int(m), int(d))
    except ValueError:
        return None


if __name__ == "__main__":
    main()

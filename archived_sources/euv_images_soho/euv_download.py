from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
from typing import List, Optional
import time

import astropy.units as u
import requests
from astropy.time import Time
from sunpy.net import Fido, attrs as a

DEFAULT_OUTPUT_DIR = Path(
    "/home/haris/Documents/ProjectBzarre/data_sources/euv_images_soho/images"
)
JSOC_SWITCH_DATE = date(2010, 6, 1)
MAX_DOWNLOAD_RETRIES = 3
RETRY_SLEEP_SECONDS = 5


def download_euv_images(
    start_date: date,
    end_date: date,
    eit_wavelength: int = 195,
    aia_wavelength: int = 193,
    output_dir: Optional[Path] = None,
) -> List[Path]:
    """
    Download SOHO/EIT FITS files between start_date and end_date (inclusive).
    """
    if start_date > end_date:
        raise ValueError("start_date must be on or before end_date.")

    output_dir = output_dir or DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: List[Path] = []
    if start_date < JSOC_SWITCH_DATE:
        legacy_end = min(end_date, JSOC_SWITCH_DATE - timedelta(days=1))
        saved_paths.extend(
            _download_vso_legacy(start_date, legacy_end, eit_wavelength, output_dir)
        )

    if end_date >= JSOC_SWITCH_DATE:
        jsoc_start = max(start_date, JSOC_SWITCH_DATE)
        saved_paths.extend(
            _download_jsoc_synoptic(jsoc_start, end_date, aia_wavelength, output_dir)
        )

    return saved_paths


def _download_vso_legacy(
    start_date: date,
    end_date: date,
    wavelength: int,
    output_dir: Path,
) -> List[Path]:
    if start_date > end_date:
        return []

    start_time = Time(datetime.combine(start_date, datetime.min.time()))
    end_time = Time(datetime.combine(end_date, datetime.max.time()))

    query = Fido.search(
        a.Time(start_time, end_time),
        a.Instrument("EIT"),
        a.Wavelength(wavelength * u.angstrom),
        a.Sample(5 * u.hour),
    )
    if query.file_num == 0:
        return []

    results = Fido.fetch(
        query,
        path=str(output_dir / "{file}"),
        overwrite=False,
        progress=False,
    )
    results = _retry_failed_parfive_results(results, "VSO legacy")
    return [Path(str(path)) for path in results]


def _download_jsoc_synoptic(
    start_date: date,
    end_date: date,
    wavelength: int,
    output_dir: Path,
) -> List[Path]:
    paths: List[Path] = []
    current = start_date
    while current <= end_date:
        for hour in range(0, 24, 6):
            file_name = f"AIA{current:%Y%m%d}_{hour:02d}00_{wavelength:04d}.fits"
            url = (
                f"http://jsoc2.stanford.edu/data/aia/synoptic/"
                f"{current:%Y/%m/%d}/H{hour:02d}00/{file_name}"
            )
            dest = output_dir / file_name
            if dest.exists():
                paths.append(dest)
                continue

            if _download_with_retries(url, dest):
                paths.append(dest)
        current += timedelta(days=1)

    return paths


def _retry_failed_parfive_results(results, label: str):
    attempt = 1
    while getattr(results, "errors", None) and results.errors and attempt < MAX_DOWNLOAD_RETRIES:
        error_count = len(results.errors)
        print(
            f"[WARN] {label} fetch had {error_count} failure(s); retrying "
            f"(attempt {attempt + 1}/{MAX_DOWNLOAD_RETRIES})..."
        )
        time.sleep(RETRY_SLEEP_SECONDS)
        results = Fido.fetch(results)
        attempt += 1

    if getattr(results, "errors", None) and results.errors:
        for err in results.errors:
            print(f"[ERROR] {label} download failed: {getattr(err, 'url', 'unknown')} -> {err.exception}")

    return results


def _download_with_retries(url: str, dest: Path) -> bool:
    attempt = 1
    while attempt <= MAX_DOWNLOAD_RETRIES:
        try:
            response = requests.get(url, timeout=120, stream=True)
            if response.status_code == 200:
                with open(dest, "wb") as f:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
                return True
            else:
                print(
                    f"[WARN] JSOC download failed ({response.status_code}) for {url}"
                )
        except requests.RequestException as exc:
            print(f"[WARN] JSOC download error for {url}: {exc}")

        attempt += 1
        if attempt <= MAX_DOWNLOAD_RETRIES:
            time.sleep(RETRY_SLEEP_SECONDS)

    print(f"[ERROR] Giving up on JSOC download after {MAX_DOWNLOAD_RETRIES} attempts: {url}")
    return False

from __future__ import annotations

from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd

import sunpy.map
import astropy.units as u
from skimage.morphology import remove_small_objects


EIT_DIR = Path("/home/haris/Documents/ProjectBzarre/preprocessing_pipeline/coronal_holes/images")
OUTPUT_CSV = Path("/home/haris/Documents/ProjectBzarre/preprocessing_pipeline/coronal_holes/ch_daily_features.csv")

CH_INTENSITY_PERCENTILE = 20.0
MIN_REGION_SIZE = 500
LOW_LAT_LIMIT = 30.0


def extract_ch_features_from_fits(path: Path) -> Dict | None:
    try:
        m = sunpy.map.Map(path)
    except Exception:
        return None

    data = m.data.astype(float)

    yy, xx = np.indices(m.data.shape)
    coords = m.pixel_to_world(xx * u.pixel, yy * u.pixel)
    r = np.sqrt(coords.Tx**2 + coords.Ty**2)
    rsun = m.rsun_obs.to(u.arcsec)
    disk_mask = r <= rsun
    data[~disk_mask] = np.nan

    median = np.nanmedian(data)
    if not np.isfinite(median) or median <= 0:
        return None
    data /= median

    threshold = np.nanpercentile(data, CH_INTENSITY_PERCENTILE)
    ch_mask = data < threshold

    ch_mask = remove_small_objects(ch_mask, max_size=max(MIN_REGION_SIZE - 1, 1))

    scale_x = m.scale[0].to(u.rad / u.pix).value
    scale_y = m.scale[1].to(u.rad / u.pix).value
    dsun_mm = m.dsun.to(u.Mm).value
    pixel_area = (scale_x * dsun_mm) * (scale_y * dsun_mm)

    disk_area = np.nansum(disk_mask) * pixel_area
    total_ch_area = np.nansum(ch_mask) * pixel_area

    yy, xx = np.indices(m.data.shape)
    coords = m.pixel_to_world(xx * u.pixel, yy * u.pixel)
    lat = coords.heliographic_stonyhurst.lat.deg

    low_lat_mask = np.abs(lat) <= LOW_LAT_LIMIT
    high_lat_mask = np.abs(lat) > LOW_LAT_LIMIT

    low_lat_area = np.nansum(ch_mask & low_lat_mask) * pixel_area
    high_lat_area = np.nansum(ch_mask & high_lat_mask) * pixel_area

    obs_time = m.date.datetime

    return {
        "timestamp": obs_time,
        "ch_area_fraction": total_ch_area / disk_area if disk_area > 0 else 0.0,
        "ch_area_lowlat_fraction": low_lat_area / disk_area if disk_area > 0 else 0.0,
        "ch_area_highlat_fraction": high_lat_area / disk_area if disk_area > 0 else 0.0,
    }


def process_all_fits(paths: List[Path]) -> pd.DataFrame:
    rows = []
    for path in paths:
        result = extract_ch_features_from_fits(path)
        if result is not None:
            rows.append(result)

    if not rows:
        raise RuntimeError("No valid EIT FITS files could be processed.")

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp")
    return df


def aggregate_daily(hourly_df: pd.DataFrame) -> pd.DataFrame:
    hourly_df["date"] = hourly_df["timestamp"].dt.floor("D")

    daily = hourly_df.groupby("date").agg(
        ch_area_mean=("ch_area_fraction", "mean"),
        ch_area_max=("ch_area_fraction", "max"),
        ch_area_lowlat_mean=("ch_area_lowlat_fraction", "mean"),
        ch_area_lowlat_max=("ch_area_lowlat_fraction", "max"),
        ch_area_highlat_mean=("ch_area_highlat_fraction", "mean"),
    )

    daily["ch_area_delta_1d"] = daily["ch_area_mean"].diff().fillna(0.0)
    daily["ch_area_rolling_3d"] = daily["ch_area_mean"].rolling(3, min_periods=1).mean()
    daily["ch_area_persistence_3d"] = (
        (daily["ch_area_mean"] > 0.05).rolling(3, min_periods=1).sum()
    )

    daily = daily.reset_index()
    return daily


def main() -> None:
    fits_paths = sorted(p for p in EIT_DIR.iterdir() if p.suffix.lower() == ".fits")
    if not fits_paths:
        raise RuntimeError(f"No FITS files found in {EIT_DIR}")

    print(f"[INFO] Processing {len(fits_paths)} EIT FITS files")

    hourly_df = process_all_fits(fits_paths)
    daily_df = aggregate_daily(hourly_df)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    daily_df.to_csv(OUTPUT_CSV, index=False)

    print(f"[OK] Daily coronal hole features written to {OUTPUT_CSV}")
    print(daily_df.head())


if __name__ == "__main__":
    main()

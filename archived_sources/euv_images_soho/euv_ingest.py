from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Dict
from io import BytesIO
import sqlite3

import sunpy.map
import numpy as np
import astropy.units as u

from space_weather_warehouse import SpaceWeatherWarehouse

CREATE_SQL = """
CREATE TABLE IF NOT EXISTS euv_images_soho (
    file_path TEXT PRIMARY KEY,
    observation_time TEXT,
    wavelength REAL,
    rsun_arcsec REAL,
    crpix1 REAL,
    crpix2 REAL,
    cdelt1 REAL,
    cdelt2 REAL,
    disk_mask BLOB,
    ch_mask BLOB
);
"""

INSERT_SQL = """
INSERT OR REPLACE INTO euv_images_soho (
    file_path,
    observation_time,
    wavelength,
    rsun_arcsec,
    crpix1,
    crpix2,
    cdelt1,
    cdelt2,
    disk_mask,
    ch_mask
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
"""


def ingest_euv_images(
    fits_files: Iterable[Path],
    warehouse: SpaceWeatherWarehouse,
    metadata: Optional[List[Dict]] = None,
) -> int:
    """
    Record downloaded EUV FITS files in SQLite for tracking.
    """
    files = list(fits_files)
    if not files:
        return 0

    warehouse.ensure_table(CREATE_SQL)
    _ensure_mask_columns(warehouse)
    rows: List[tuple] = []
    if metadata is not None:
        for record in metadata:
            if "disk_mask" not in record or "ch_mask" not in record:
                try:
                    extra = _extract_spoca_raw_data(sunpy.map.Map(record["file_path"]))
                except Exception:
                    continue
                record = {**record, **extra}
            rows.append(
                (
                    record["file_path"],
                    record["observation_time"],
                    record["wavelength"],
                    record["rsun_arcsec"],
                    record["crpix1"],
                    record["crpix2"],
                    record["cdelt1"],
                    record["cdelt2"],
                    record["disk_mask"],
                    record["ch_mask"],
                )
            )
    else:
        for path in files:
            try:
                m = sunpy.map.Map(path)
            except Exception:
                continue

            record = _extract_spoca_raw_data(m)
            rows.append(
                (
                    str(path),
                    m.date.isot,
                    float(m.wavelength.value),
                    record["rsun_arcsec"],
                    record["crpix1"],
                    record["crpix2"],
                    record["cdelt1"],
                    record["cdelt2"],
                    record["disk_mask"],
                    record["ch_mask"],
                )
            )

    if not rows:
        return 0

    return warehouse.insert_rows(INSERT_SQL, rows)


def _extract_spoca_raw_data(m: sunpy.map.GenericMap) -> dict:
    """
    Extract basic disk / coronal hole proxy masks and geometry metadata.

    This is a lightweight percentile-based segmentation, not the full SPOCA pipeline.
    """
    data = m.data.astype("float32").copy()

    yy, xx = np.indices(data.shape)
    coords = m.pixel_to_world(xx * u.pixel, yy * u.pixel)
    r = np.sqrt(coords.Tx**2 + coords.Ty**2)
    rsun = m.rsun_obs.to(u.arcsec)
    disk_mask = r <= rsun
    data[~disk_mask] = np.nan

    median = np.nanmedian(data)
    if not np.isfinite(median) or median <= 0:
        median = 1.0
    normalized = data / median
    finite_mask = np.isfinite(normalized)
    if finite_mask.any():
        threshold = np.nanpercentile(normalized[finite_mask], 20.0)
    else:
        threshold = 0.0
    ch_mask = finite_mask & (normalized < threshold)

    return {
        "rsun_arcsec": float(rsun.value),
        "crpix1": float(m.reference_pixel.x.value),
        "crpix2": float(m.reference_pixel.y.value),
        "cdelt1": float(m.scale[0].value),
        "cdelt2": float(m.scale[1].value),
        "disk_mask": _serialize_mask(disk_mask),
        "ch_mask": _serialize_mask(ch_mask),
    }


def _serialize_mask(mask: np.ndarray) -> bytes:
    buffer = BytesIO()
    np.save(buffer, mask.astype(np.uint8), allow_pickle=False)
    return buffer.getvalue()


def _ensure_mask_columns(warehouse: SpaceWeatherWarehouse) -> None:
    with sqlite3.connect(warehouse.db_path) as conn:
        cursor = conn.execute("PRAGMA table_info(euv_images_soho)")
        columns = {row[1] for row in cursor.fetchall()}

    ddl_statements = []
    if "disk_mask" not in columns:
        ddl_statements.append("ALTER TABLE euv_images_soho ADD COLUMN disk_mask BLOB;")
    if "ch_mask" not in columns:
        ddl_statements.append("ALTER TABLE euv_images_soho ADD COLUMN ch_mask BLOB;")

    if ddl_statements:
        with sqlite3.connect(warehouse.db_path) as conn:
            for ddl in ddl_statements:
                conn.execute(ddl)
            conn.commit()

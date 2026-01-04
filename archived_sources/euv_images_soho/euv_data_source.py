from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Iterable, List, Dict
import pandas as pd
import sunpy.map

from space_weather_api import SpaceWeatherAPI
from space_weather_warehouse import SpaceWeatherWarehouse

from data_sources.euv_images_soho.euv_download import download_euv_images, DEFAULT_OUTPUT_DIR
import astropy.units as u
import sunpy.map

from data_sources.euv_images_soho.euv_ingest import ingest_euv_images, _extract_spoca_raw_data
from data_sources.euv_images_soho.euv_plot import plot_euv_metadata


class EUVImagesSOHODataSource(SpaceWeatherAPI):
    """
    Access SOHO/EIT EUV imagery via the shared download/ingest/plot interface.
    """
    RUN_IN_THREADPOOL = False

    def __init__(
        self,
        days,
        output_dir: Path | None = None,
    ) -> None:
        super().__init__(days)
        self.output_dir = output_dir or DEFAULT_OUTPUT_DIR
        self._latest_metadata: Dict[str, Dict] = {}

    def _download_impl(self) -> list[Path]:
        return download_euv_images(self.start_date, self.end_date, output_dir=self.output_dir)

    def ingest(
        self,
        files: Iterable[Path],
        warehouse: SpaceWeatherWarehouse | None = None,
        db_path: str = "space_weather.db",
    ):
        warehouse = warehouse or SpaceWeatherWarehouse(db_path)
        file_list = list(files)
        metadata = self._prepare_metadata(file_list)
        inserted = ingest_euv_images(file_list, warehouse, metadata=metadata)
        if file_list:
            _cleanup_files(file_list)
        self._latest_metadata.clear()
        return inserted

    def download_ingest_cleanup(
        self,
        batch_size: int = 40,
        warehouse: SpaceWeatherWarehouse | None = None,
        db_path: str = "space_weather.db",
    ) -> dict[str, int]:
        """
        Stream downloads day-by-day, ingest in batches, and delete FITS files to conserve disk space.
        """
        warehouse = warehouse or SpaceWeatherWarehouse(db_path)
        batch: list[Path] = []
        total_downloaded = 0
        total_inserted = 0

        for single_day in self.iter_days():
            files = download_euv_images(single_day, single_day, output_dir=self.output_dir)
            if not files:
                continue

            batch.extend(files)
            total_downloaded += len(files)

            while len(batch) >= batch_size:
                chunk = batch[:batch_size]
                total_inserted += ingest_euv_images(chunk, warehouse)
                _cleanup_files(chunk)
                batch = batch[batch_size:]

        if batch:
            total_inserted += ingest_euv_images(batch, warehouse)
            _cleanup_files(batch)
            batch.clear()

        return {"downloaded": total_downloaded, "inserted": total_inserted}

    def plot(self, files: Iterable[Path]) -> None:
        files = list(files)
        if not files:
            raise ValueError("No FITS files provided to plot.")
        # Recompute metadata from the first file for visualization
        path = files[0]
        m = sunpy.map.Map(path)
        metadata = {
            "rsun_arcsec": float(m.rsun_obs.to(u.arcsec).value),
            "crpix1": float(m.reference_pixel.x.value),
            "crpix2": float(m.reference_pixel.y.value),
            "cdelt1": float(m.scale[0].value),
            "cdelt2": float(m.scale[1].value),
        }
        plot_euv_metadata(metadata)

    def tracker_payload(self, data):
        if not isinstance(data, list):
            return data
        metadata = self._prepare_metadata(data)
        timestamps = [
            pd.to_datetime(record["observation_time"])
            for record in metadata
            if record.get("observation_time")
        ]
        if not timestamps:
            return pd.DataFrame(columns=["observation_time"])
        return pd.DataFrame({"observation_time": timestamps})

    def _prepare_metadata(self, file_paths: Iterable[Path]) -> List[Dict]:
        metadata: List[Dict] = []
        for path in file_paths:
            key = str(path)
            if key in self._latest_metadata:
                metadata.append(self._latest_metadata[key])
                continue
            record = _extract_metadata_from_file(path)
            if record:
                metadata.append(record)
                self._latest_metadata[key] = record
        return metadata


def _cleanup_files(paths: Iterable[Path]) -> None:
    for path in paths:
        try:
            path.unlink()
        except FileNotFoundError:
            continue


def _extract_metadata_from_file(path: Path) -> Dict | None:
    try:
        m = sunpy.map.Map(path)
    except Exception:
        return None

    record = _extract_spoca_raw_data(m)
    return {
        "file_path": str(path),
        "observation_time": m.date.isot,
        "wavelength": float(m.wavelength.value),
        "rsun_arcsec": record["rsun_arcsec"],
        "crpix1": record["crpix1"],
        "crpix2": record["crpix2"],
        "cdelt1": record["cdelt1"],
        "cdelt2": record["cdelt2"],
    }

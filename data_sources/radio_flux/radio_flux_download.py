from datetime import date, timedelta
from typing import List, Optional
from pathlib import Path
import sys

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.http import http_get

RADIO_FLUX_NEW_URL = "https://www.spaceweather.gc.ca/solar_flux_data/daily_flux_values/fluxtable.txt"
RADIO_FLUX_OLD_URL = "https://www.spaceweather.gc.ca/solar_flux_data/daily_flux_values/F107_1996_2007.txt"
RADIO_FLUX_COLUMNS = ["time_tag", "observed_flux", "adjusted_flux", "ursi_flux"]
OLD_DATA_START = date(1996, 2, 14)
OLD_DATA_END = date(2004, 10, 27)
NEW_DATA_START = OLD_DATA_END + timedelta(days=1)


def download_radio_flux(
    start_date: date, end_date: date, session: Optional[requests.Session] = None
) -> pd.DataFrame:
    """
    Download Penticton F10.7 radio flux for the requested date range.
    """
    if start_date > end_date:
        raise ValueError("start_date must be on or before end_date.")

    session = session or requests.Session()
    frames: List[pd.DataFrame] = []

    if start_date <= OLD_DATA_END and end_date >= OLD_DATA_START:
        response_old = http_get(
            RADIO_FLUX_OLD_URL, session=session, log_name="Radio Flux (legacy)", timeout=30
        )
        if response_old is not None:
            df_old = _parse_flux_table(response_old.text)
            if not df_old.empty:
                mask_old = (
                    (df_old["time_tag"].dt.date >= max(start_date, OLD_DATA_START))
                    & (df_old["time_tag"].dt.date <= min(end_date, OLD_DATA_END))
                )
                frames.append(df_old.loc[mask_old])

    response_new = http_get(
        RADIO_FLUX_NEW_URL, session=session, log_name="Radio Flux", timeout=30
    )
    if response_new is not None:
        df_new = _parse_flux_table(response_new.text)
        mask_new = (df_new["time_tag"].dt.date >= start_date) & (
            df_new["time_tag"].dt.date <= end_date
        )
        frames.append(df_new.loc[mask_new])

    if not frames:
        return pd.DataFrame(columns=RADIO_FLUX_COLUMNS)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values("time_tag").drop_duplicates("time_tag", keep="last")
    combined = combined.reset_index(drop=True)

    mask = (combined["time_tag"].dt.date >= start_date) & (
        combined["time_tag"].dt.date <= end_date
    )
    combined = combined.loc[mask].reset_index(drop=True)
    return combined.reindex(columns=RADIO_FLUX_COLUMNS)


def _parse_flux_table(text: str) -> pd.DataFrame:
    records = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("fluxdate") or stripped.startswith("-----"):
            continue
        parts = stripped.split()
        if len(parts) < 7:
            continue
        record = {
            "fluxdate": parts[0],
            "fluxtime": parts[1],
            "fluxobsflux": parts[4],
            "fluxadjflux": parts[5],
            "fluxursi": parts[6],
        }
        records.append(record)

    if not records:
        return pd.DataFrame(columns=RADIO_FLUX_COLUMNS)

    df = pd.DataFrame(records)
    df["time_tag"] = pd.to_datetime(
        df["fluxdate"].astype(str).str.zfill(8)
        + df["fluxtime"].astype(str).str.zfill(6),
        format="%Y%m%d%H%M%S",
        errors="coerce",
    )
    df["observed_flux"] = pd.to_numeric(df["fluxobsflux"], errors="coerce")
    df["adjusted_flux"] = pd.to_numeric(df["fluxadjflux"], errors="coerce")
    df["ursi_flux"] = pd.to_numeric(df["fluxursi"], errors="coerce")

    df = df.dropna(subset=["time_tag"])
    return df[["time_tag", "observed_flux", "adjusted_flux", "ursi_flux"]]

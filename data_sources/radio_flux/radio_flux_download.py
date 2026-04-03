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

# Current rolling flux table published by the Canadian Space Weather Forecast Centre
RADIO_FLUX_NEW_URL = "https://www.spaceweather.gc.ca/solar_flux_data/daily_flux_values/fluxtable.txt"
# Separate archive file covering the legacy period before the rolling table was maintained
RADIO_FLUX_OLD_URL = "https://www.spaceweather.gc.ca/solar_flux_data/daily_flux_values/F107_1996_2007.txt"
# Output column order expected by downstream ingest and plot steps
RADIO_FLUX_COLUMNS = ["time_tag", "observed_flux", "adjusted_flux", "ursi_flux"]
# Inclusive date boundaries of the legacy archive file
OLD_DATA_START = date(1996, 2, 14)
OLD_DATA_END = date(2004, 10, 27)
# First date covered exclusively by the new rolling table
NEW_DATA_START = OLD_DATA_END + timedelta(days=1)


def download_radio_flux(
    start_date: date, end_date: date, session: Optional[requests.Session] = None
) -> pd.DataFrame:
    """
    Download Penticton F10.7 radio flux for the requested date range.
    """
    if start_date > end_date:
        raise ValueError("start_date must be on or before end_date.")

    # Reuse a single session for all HTTP calls to benefit from connection pooling
    session = session or requests.Session()
    frames: List[pd.DataFrame] = []

    # Only fetch the legacy archive when the requested range overlaps with it
    if start_date <= OLD_DATA_END and end_date >= OLD_DATA_START:
        response_old = http_get(
            RADIO_FLUX_OLD_URL, session=session, log_name="Radio Flux (legacy)", timeout=30
        )
        if response_old is not None:
            df_old = _parse_flux_table(response_old.text)
            if not df_old.empty:
                # Clip to the intersection of the requested range and the archive's coverage
                mask_old = (
                    (df_old["time_tag"].dt.date >= max(start_date, OLD_DATA_START))
                    & (df_old["time_tag"].dt.date <= min(end_date, OLD_DATA_END))
                )
                frames.append(df_old.loc[mask_old])

    # Always fetch the rolling table; it may contain rows that overlap with the archive
    response_new = http_get(
        RADIO_FLUX_NEW_URL, session=session, log_name="Radio Flux", timeout=30
    )
    if response_new is not None:
        df_new = _parse_flux_table(response_new.text)
        mask_new = (df_new["time_tag"].dt.date >= start_date) & (
            df_new["time_tag"].dt.date <= end_date
        )
        frames.append(df_new.loc[mask_new])

    # Return an empty frame with the correct schema if neither source yielded data
    if not frames:
        return pd.DataFrame(columns=RADIO_FLUX_COLUMNS)

    combined = pd.concat(frames, ignore_index=True)
    # Sort chronologically and resolve any overlap between the two sources, preferring the newer file
    combined = combined.sort_values("time_tag").drop_duplicates("time_tag", keep="last")
    combined = combined.reset_index(drop=True)

    # Final clip to the exact requested range after deduplication
    mask = (combined["time_tag"].dt.date >= start_date) & (
        combined["time_tag"].dt.date <= end_date
    )
    combined = combined.loc[mask].reset_index(drop=True)
    return combined.reindex(columns=RADIO_FLUX_COLUMNS)


def _parse_flux_table(text: str) -> pd.DataFrame:
    records = []
    for line in text.splitlines():
        stripped = line.strip()
        # Skip blank lines and known header / separator patterns present in both file formats
        if (
            not stripped
            or stripped.startswith("Julian")
            or stripped.startswith("Number")
            or stripped.startswith("=")
            or stripped.lower().startswith("fluxdate")
            or stripped.startswith("-----")
        ):
            continue
        parts = stripped.split()
        # New rolling table format: 7 whitespace-separated columns
        if len(parts) == 7:
            date_token = parts[0]
            time_token = parts[1]
            obs, adj, ursi = parts[4], parts[5], parts[6]
        # Legacy archive format: 9+ columns with the date spread across three fields
        elif len(parts) >= 9:
            date_token = f"{parts[2]}{parts[3]}{parts[4]}"
            time_token = parts[5]
            obs, adj, ursi = parts[6], parts[7], parts[8]
        else:
            # Unrecognised row shape — skip silently
            continue

        # Normalise HHMM tokens to HHMMSS so strptime can parse them uniformly
        if len(time_token) <= 4:
            time_token = time_token + "00"
        time_token = time_token.zfill(6)

        records.append(
            {
                "time_str": date_token + time_token,
                "fluxobsflux": obs,
                "fluxadjflux": adj,
                "fluxursi": ursi,
            }
        )

    if not records:
        return pd.DataFrame(columns=RADIO_FLUX_COLUMNS)

    df = pd.DataFrame(records)
    # Parse the concatenated datetime string into a proper Timestamp; unparseable rows become NaT
    df["time_tag"] = pd.to_datetime(df["time_str"], format="%Y%m%d%H%M%S", errors="coerce")
    # Coerce flux strings to float; non-numeric sentinel values (e.g. "-1") become NaN
    df["observed_flux"] = pd.to_numeric(df["fluxobsflux"], errors="coerce")
    df["adjusted_flux"] = pd.to_numeric(df["fluxadjflux"], errors="coerce")
    df["ursi_flux"] = pd.to_numeric(df["fluxursi"], errors="coerce")

    # Drop rows where the timestamp could not be parsed — they cannot be placed on a timeline
    df = df.dropna(subset=["time_tag"])
    return df[["time_tag", "observed_flux", "adjusted_flux", "ursi_flux"]]

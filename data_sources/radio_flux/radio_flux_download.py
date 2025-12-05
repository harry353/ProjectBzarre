from datetime import date
from io import StringIO
from typing import Optional

import pandas as pd
import requests

from common.http import http_get

RADIO_FLUX_URL = "https://www.spaceweather.gc.ca/solar_flux_data/daily_flux_values/fluxtable.txt"
RADIO_FLUX_COLUMNS = ["time_tag", "observed_flux", "adjusted_flux", "ursi_flux"]


def download_radio_flux(
    start_date: date, end_date: date, session: Optional[requests.Session] = None
) -> pd.DataFrame:
    """
    Download Penticton F10.7 radio flux for the requested date range.
    """
    if start_date > end_date:
        raise ValueError("start_date must be on or before end_date.")

    session = session or requests.Session()
    response = http_get(
        RADIO_FLUX_URL, session=session, log_name="Radio Flux", timeout=30
    )
    if response is None:
        return pd.DataFrame(columns=RADIO_FLUX_COLUMNS)

    df = _parse_flux_table(response.text)
    if df.empty:
        return df

    mask = (df["time_tag"].dt.date >= start_date) & (
        df["time_tag"].dt.date <= end_date
    )
    df = df.loc[mask].reset_index(drop=True)
    return df.reindex(columns=RADIO_FLUX_COLUMNS)


def _parse_flux_table(text: str) -> pd.DataFrame:
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("fluxdate") or stripped.startswith("-----"):
            continue
        lines.append(stripped)

    if not lines:
        return pd.DataFrame(columns=RADIO_FLUX_COLUMNS)

    data = "\n".join(lines)
    df = pd.read_csv(
        StringIO(data),
        sep=r"\s+",
        names=[
            "fluxdate",
            "fluxtime",
            "fluxjulian",
            "fluxcarrington",
            "fluxobsflux",
            "fluxadjflux",
            "fluxursi",
        ],
    )

    df["time_tag"] = pd.to_datetime(
        df["fluxdate"].astype(str) + df["fluxtime"].astype(str).str.zfill(6),
        format="%Y%m%d%H%M%S",
        errors="coerce",
    )
    for column in ["fluxobsflux", "fluxadjflux", "fluxursi"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=["time_tag"])
    df.rename(
        columns={
            "fluxobsflux": "observed_flux",
            "fluxadjflux": "adjusted_flux",
            "fluxursi": "ursi_flux",
        },
        inplace=True,
    )
    return df[["time_tag", "observed_flux", "adjusted_flux", "ursi_flux"]]

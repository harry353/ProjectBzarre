import pandas as pd

from common.http import http_get

DONKI_FLARE_URL = "https://kauai.ccmc.gsfc.nasa.gov/DONKI/WS/get/FLR"
VALID_CLASSES = {"A", "B", "C", "M", "X"}
TIME_COLUMNS = ("beginTime", "peakTime", "endTime", "submissionTime")


def download_flares(start_date, end_date):
    """
    Fetch NASA DONKI solar flare events for the provided date range.
    """
    params = {
        "startDate": start_date.isoformat(),
        "endDate": end_date.isoformat(),
        "catalog": "M2M_CATALOG",
        "class": "ALL",
        "version": "Latest",
    }

    response = http_get(DONKI_FLARE_URL, params=params, timeout=30, log_name="Solar Flare")
    if response is None:
        return pd.DataFrame()

    data = response.json()

    df = pd.DataFrame(data)
    if df.empty:
        return df

    for column in TIME_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_datetime(df[column], errors="coerce", utc=True)

    if "classType" in df.columns:
        df["classLetter"] = df["classType"].astype(str).str[0].str.upper()
    else:
        df["classLetter"] = ""
    df = df[df["classLetter"].isin(VALID_CLASSES)]

    return df.reset_index(drop=True)

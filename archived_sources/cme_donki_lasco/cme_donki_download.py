import pandas as pd

from common.http import http_get

DONKI_CME_URL = "https://kauai.ccmc.gsfc.nasa.gov/DONKI/WS/get/CMEAnalysis"


def download_cme(start_date, end_date):
    """
    Retrieve CMEAnalysis entries for the provided date range.

    Parameters
    ----------
    start_date : datetime.date
    end_date : datetime.date
    use_sample_on_failure : bool, optional
        Load bundled sample data when the remote API is unavailable.
    sample_path : pathlib.Path, optional
        Path to the local sample dataset.

    Returns
    -------
    pandas.DataFrame
        Cleaned CME entries filtered to the requested date range.
    """

    params = {
        "startDate": start_date.isoformat(),
        "endDate": end_date.isoformat(),
        "mostAccurateOnly": "true",
        "completeEntryOnly": "true",
        "speed": 0,
        "halfAngle": 0,
        "catalog": "ALL",
    }

    response = http_get(DONKI_CME_URL, params=params, timeout=30, log_name="CME")
    if response is None:
        return pd.DataFrame()

    data = response.json()
    df = pd.DataFrame(data)
    if df.empty:
        return df

    if "time21_5" in df.columns:
        df["time21_5"] = pd.to_datetime(df["time21_5"], errors="coerce", utc=True)
        df = df.dropna(subset=["time21_5"])
        mask = (
            (df["time21_5"].dt.date >= start_date)
            & (df["time21_5"].dt.date <= end_date)
        )
        df = df.loc[mask]

    return df.reset_index(drop=True)

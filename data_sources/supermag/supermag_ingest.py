import pandas as pd

from space_weather_warehouse import SpaceWeatherWarehouse

SUPERMAG_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS supermag_indices (
    time_tag TEXT PRIMARY KEY,
    sml REAL,
    smu REAL,
    sme REAL,
    smo REAL
);
"""

SUPERMAG_INSERT_SQL = """
INSERT OR REPLACE INTO supermag_indices (time_tag, sml, smu, sme, smo)
VALUES (?, ?, ?, ?, ?);
"""

SUPERMAG_COLUMNS = ["time", "SML", "SMU", "SME", "SMO"]


def ingest_supermag(df: pd.DataFrame, warehouse: SpaceWeatherWarehouse):
    """
    Write SuperMAG indices into SQLite via SpaceWeatherWarehouse.
    """
    if df.empty:
        return 0

    warehouse.ensure_table(SUPERMAG_TABLE_SQL)
    payload = _prepare_payload(df)

    rows = []
    for _, row in payload.iterrows():
        rows.append(
            (
                row["time"].isoformat(),
                _to_float(row.get("SML")),
                _to_float(row.get("SMU")),
                _to_float(row.get("SME")),
                _to_float(row.get("SMO")),
            )
        )

    return warehouse.insert_rows(SUPERMAG_INSERT_SQL, rows)


def _prepare_payload(df: pd.DataFrame) -> pd.DataFrame:
    payload = df.copy()

    if "time" in payload:
        payload["time"] = pd.to_datetime(payload["time"], errors="coerce")
    elif "time_tag" in payload:
        payload["time"] = pd.to_datetime(payload["time_tag"], errors="coerce")
    elif "tval" in payload:
        payload["time"] = pd.to_datetime(payload["tval"], unit="s", errors="coerce")
    else:
        payload["time"] = pd.NaT

    payload = payload.dropna(subset=["time"]).sort_values("time")

    if "SME" not in payload or payload["SME"].isna().all():
        if "SMU" in payload and "SML" in payload:
            payload["SME"] = payload["SMU"] - payload["SML"]
        else:
            payload["SME"] = pd.NA

    if "SMO" not in payload or payload["SMO"].isna().all():
        if "SMU" in payload and "SML" in payload:
            payload["SMO"] = (payload["SMU"] + payload["SML"]) / 2
        else:
            payload["SMO"] = pd.NA

    for column in ["SML", "SMU"]:
        if column not in payload:
            payload[column] = pd.NA

    return payload.reindex(columns=SUPERMAG_COLUMNS)


def _to_float(value):
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None

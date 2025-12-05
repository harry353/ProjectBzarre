from space_weather_warehouse import SpaceWeatherWarehouse

RADIO_FLUX_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS radio_flux (
    time_tag TEXT PRIMARY KEY,
    observed_flux REAL,
    adjusted_flux REAL,
    ursi_flux REAL
);
"""

RADIO_FLUX_INSERT_SQL = """
INSERT OR REPLACE INTO radio_flux (time_tag, observed_flux, adjusted_flux, ursi_flux)
VALUES (?, ?, ?, ?);
"""


def ingest_radio_flux(df, warehouse: SpaceWeatherWarehouse):
    """
    Persist radio flux readings into SQLite.
    """
    if df.empty:
        return 0

    warehouse.ensure_table(RADIO_FLUX_TABLE_SQL)
    payload = df.copy().reindex(
        columns=["time_tag", "observed_flux", "adjusted_flux", "ursi_flux"]
    )
    payload["time_tag"] = payload["time_tag"].astype(str)

    rows = []
    for _, row in payload.iterrows():
        rows.append(
            (
                row["time_tag"],
                _to_float(row["observed_flux"]),
                _to_float(row["adjusted_flux"]),
                _to_float(row["ursi_flux"]),
            )
        )

    return warehouse.insert_rows(RADIO_FLUX_INSERT_SQL, rows)


def _to_float(value):
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


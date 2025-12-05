import numpy as np

XRAY_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS xray_flux (
    time_tag TEXT PRIMARY KEY,
    irradiance_xrsa1 REAL,
    irradiance_xrsa2 REAL,
    irradiance_xrsb1 REAL,
    irradiance_xrsb2 REAL,
    xrs_ratio REAL
);
"""

XRAY_INSERT_SQL = """
INSERT OR REPLACE INTO xray_flux
(time_tag, irradiance_xrsa1, irradiance_xrsa2, irradiance_xrsb1, irradiance_xrsb2, xrs_ratio)
VALUES (?, ?, ?, ?, ?, ?);
"""


def ingest_xrs_goes(df, warehouse):
    """
    Insert GOES XRS rows into SQLite.
    """

    if df.empty:
        return 0

    warehouse.ensure_table(XRAY_TABLE_SQL)

    rows = []
    serialised = df.reset_index().rename(columns={"index": "time_tag"})

    for _, row in serialised.iterrows():
        rows.append([
            _py(row["time_tag"].isoformat() if row.get("time_tag") else None),
            _py(row.get("irradiance_xrsa1")),
            _py(row.get("irradiance_xrsa2")),
            _py(row.get("irradiance_xrsb1")),
            _py(row.get("irradiance_xrsb2")),
            _py(row.get("xrs_ratio")),
        ])

    return warehouse.insert_rows(XRAY_INSERT_SQL, rows)


def _py(x):
    """
    Convert NumPy numeric types into Python native types.
    """
    if x is None:
        return None
    if isinstance(x, (np.floating, np.float32, np.float64)):
        return float(x)
    if isinstance(x, (np.integer, np.int32, np.int64)):
        return int(x)
    return x

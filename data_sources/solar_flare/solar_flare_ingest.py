from space_weather_warehouse import SpaceWeatherWarehouse

FLARE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS flare_events (
    flrID TEXT PRIMARY KEY,
    beginTime TEXT,
    peakTime TEXT,
    endTime TEXT,
    classType TEXT,
    classLetter TEXT,
    sourceLocation TEXT,
    activeRegionNum INTEGER,
    note TEXT,
    submissionTime TEXT,
    versionId TEXT,
    link TEXT
);
"""

FLARE_INSERT_SQL = """
INSERT OR IGNORE INTO flare_events
(flrID, beginTime, peakTime, endTime, classType, classLetter,
 sourceLocation, activeRegionNum, note, submissionTime, versionId, link)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
"""

FLARE_COLUMNS = [
    "flrID",
    "beginTime",
    "peakTime",
    "endTime",
    "classType",
    "classLetter",
    "sourceLocation",
    "activeRegionNum",
    "note",
    "submissionTime",
    "versionId",
    "link",
]


def ingest_flares(df, warehouse: SpaceWeatherWarehouse):
    """
    Insert flare rows into SQLite using the provided warehouse.
    """
    if df.empty:
        return 0

    warehouse.ensure_table(FLARE_TABLE_SQL)

    payload = df.copy()
    for column in ("beginTime", "peakTime", "endTime", "submissionTime"):
        if column in payload.columns:
            payload[column] = payload[column].astype(str)

    payload = payload[FLARE_COLUMNS]
    payload = payload.drop_duplicates(subset=["flrID"])

    return warehouse.insert_rows(FLARE_INSERT_SQL, payload.values.tolist())

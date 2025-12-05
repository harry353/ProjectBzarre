# GitHub Copilot instructions for this repository

Purpose: give an AI coding agent the minimal, actionable knowledge to be immediately productive in this codebase.

Big picture
- Architecture: top-level orchestrator is `regenerate_db.py`. It discovers data-source modules under `data_sources/`, instantiates classes that subclass `SpaceWeatherAPI`, then calls `download()`, `ingest(...)`, and (optionally) `plot()`.
- Core helpers: `space_weather_api.py` (base class and `days` parsing), `space_weather_warehouse.py` (lightweight SQLite helper). Data sources live in `data_sources/<name>/` and follow a 3-file convention: `*_download.py`, `*_ingest.py`, `*_plot.py` and a `*_data_source.py` that ties them together.

Data flow and conventions
- Discovery: `regenerate_db.iter_data_source_modules()` looks for modules ending with `_data_source` using `pkgutil.walk_packages` rooted at `data_sources/`.
- Data-source contract: subclasses of `SpaceWeatherAPI` must implement `download() -> pandas.DataFrame`, `ingest(df, warehouse=None, db_path='space_weather.db') -> int`, and `plot(df)`.
- `days` argument formats (see `SpaceWeatherAPI._parse_days_argument`):
  - integer n: last `n` days (today - n + 1 -> today)
  - single `date`: that date -> today
  - `(date, date)`: explicit start and end
  - `(date, timedelta)`: start + duration
- SQLite pattern: each `*_ingest.py` provides a DDL string (e.g. `CME_TABLE_SQL`) and an INSERT SQL (e.g. `CME_INSERT_SQL`). Use `SpaceWeatherWarehouse.ensure_table(DDL)` then `warehouse.insert_rows(SQL, rows)` to persist.
- Ingest helpers: ingest functions commonly accept a `warehouse` instance OR a `db_path` string. `regenerate_db.process_sources()` constructs a `SpaceWeatherWarehouse` and passes it to `source.ingest(df, warehouse=warehouse)` — prefer that approach for bulk ingestion.
- idempotency: ingest SQL mostly uses `INSERT OR IGNORE` and ingest code de-duplicates rows (`drop_duplicates`) before insert.

Patterns to follow when adding a new data source
- Directory layout: `data_sources/<newname>/`
  - `<newname>_data_source.py` — subclass of `SpaceWeatherAPI` that delegates to the other modules.
  - `<newname>_download.py` — `download_<name>(start_date, end_date) -> pandas.DataFrame` (return empty DataFrame on failure).
  - `<newname>_ingest.py` — DDL string, INSERT SQL, columns list, and `ingest_<name>(df, warehouse)` that calls `warehouse.ensure_table()` and `warehouse.insert_rows()`.
  - `<newname>_plot.py` — plotting helpers; note plots call `matplotlib.pyplot.show()` and will block in interactive mode.

Developer workflows and commands
- Run the full DB regeneration locally (resets `space_weather.db`):
```
python regenerate_db.py
```
- Run a single data-source test script (these live in `tests/` and are runnable scripts):
```
python tests/test_cme.py
python tests/test_imf.py
```
- Run tests with pytest (plots may block in headless CI):
```
export MATPLOTLIB_BACKEND=Agg  # use in CI or headless runs
python -m pytest -q
```
- Virtualenv + install (recommended):
```
python -m venv .venv
source .venv/bin/activate
pip install pandas numpy requests matplotlib pytest
```

Important implementation notes discovered in the code
- Download functions generally return an empty `pandas.DataFrame` on failure (see `data_sources/*/*_download.py`). callers should handle empty DataFrames gracefully.
- Ingestors convert timestamps to strings before inserting into SQLite (e.g. `payload['time21_5'] = payload['time21_5'].astype(str)`). Expect datetimes to be persisted as text.
- Plots assume `pandas` datetime types and group by `df["time..."].dt.date`. They raise on empty dataframes.
- The discovery mechanism only loads modules whose file/module name ends with `_data_source`. Name your module accordingly so `regenerate_db` finds it.

Integration & external dependencies
- External HTTP APIs: e.g. CME uses NASA DONKI (`DONKI_CME_URL`) via `requests` with a 30s timeout; failures are logged and an empty DataFrame returned.
- Persistence: lightweight SQLite at `space_weather.db` by default. All schema management is done in ingest modules via `ensure_table(DDL)`.

Examples (explicit references)
- Orchestration and discovery: `regenerate_db.py::iter_data_source_modules()` and `regenerate_db.py::process_sources()`.
- Base API contract and `days` parsing: `space_weather_api.py`.
- SQLite helper: `space_weather_warehouse.py`.
- Concrete data source example: `data_sources/cme/cme_data_source.py` delegates to `cme_download.py`, `cme_ingest.py`, and `cme_plot.py`.

What the AI agent should do first when contributing
- Run the tests locally (use `MATPLOTLIB_BACKEND=Agg` in headless environments).
- Add new data sources by following the 4-file convention and naming the data-source module `<name>_data_source.py`.
- Keep ingestion idempotent: dedupe rows, use `INSERT OR IGNORE`, and supply DDL in the ingest module.

If anything here is unclear or you want additional examples (CI config, dependency pinning, or a contributor checklist), tell me which area to expand and I'll update this file.

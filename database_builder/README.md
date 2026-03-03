# Database Builder

This directory provides the orchestration framework for fetching, maintaining, and scaling the ingestion of the raw space weather SQLite warehouse (`space_weather.db`).

## Key Components

- **`create_db.py` (Main Entrypoint)**: The primary orchestration script. It initializes database regeneration over user-defined or default temporal boundaries (e.g., from 2005 onwards).
- **`processor.py`**: The parallel execution engine. It dynamically handles the retrieval of data sources utilizing Python's `concurrent.futures.ThreadPoolExecutor`. Also employs throttling locks to respect API rate limits from different data providers.
- **`tracker.py`**: A persistent state management tool that writes back to a CSV log. It ensures that the database orchestration remembers the latest timestamps fetched for each data source provider—preventing duplicate download cycles if a job is halted or resumed sequentially.
- **`discovery.py` & `helpers.py`**: Utility scripts used by the orchestrator to dynamically locate and instantiate the various `SpaceWeatherAPI` sub-classes located inside the `data_sources` directory structure.

## Execution Example

To invoke the database rebuild process, execute `create_db.py` from the root of the project:

```bash
# Build the warehouse from 2005-01-01 to 2024-01-01
python3 database_builder/create_db.py --start 2005-01-01 --end 2024-01-01
```

*(If no arguments are provided, it defaults to the temporal extents specified in `database_builder/constants.py`)*

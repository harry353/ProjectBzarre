# Common Utilities

This directory contains shared helper functions and core utilities used across the various `ProjectBzarre` pipelines (data downloading, preprocessing, and model inference). Keeping these utilities centralized ensures consistent logging and network handling behaviors project-wide.

## Key Utilities

### `logging.py`
Provides a project-wide logging handler that intercepts the standard `print` function. It parses the log messages and seamlessly injects colored terminal output based on severity labels.

**Supported Labels:**
- `[INFO]` (Blue)
- `[OK]` (Green)
- `[WARN]` / `[WARNING]` (Yellow)
- `[ERROR]` (Red)
- `[SKIP]` (Cyan)

### `http.py`
A resilient wrapper around the `requests.get` library for reliable data downloading. It offers:
- Automatic request retries with programmable delays to handle intermittent network issues.
- Integrated, formatted logging that leverages `logging.py`.
- Thread-local session management (via `THREAD_SESSION`) to optimize performance when executing pipelines concurrently.

from typing import Iterable, Optional
import time
import threading

import requests

# Thread-local storage used to share a per-thread requests.Session across callers.
THREAD_SESSION = threading.local()


# Global in-memory cache for GET requests to prevent redundant network hits across threads
_GET_CACHE: dict[str, requests.Response | None] = {}
# Lock protecting all reads and writes to _GET_CACHE.
_CACHE_LOCK = threading.Lock()


def http_get(
    url: str,
    *,
    session: Optional[requests.Session] = None,
    log_name: Optional[str] = None,
    allowed_statuses: Optional[Iterable[int]] = None,
    raise_for_status: bool = True,
    retries: int = 3,
    retry_delay: float = 1.0,
    use_cache: bool = False,
    **kwargs,
) -> Optional[requests.Response]:
    """
    Issue an HTTP GET request with consistent logging and optional caching.

    If use_cache is True, successful responses and permanent failures (None)
    are cached by URL for the lifetime of the process.
    """
    # Return cached result immediately if available, avoiding a network round-trip.
    if use_cache:
        with _CACHE_LOCK:
            if url in _GET_CACHE:
                return _GET_CACHE[url]

    allowed = set(allowed_statuses or [])
    # Prefer the explicitly passed session, then the thread-local one, then a fresh session.
    client = session or getattr(THREAD_SESSION, "session", None) or requests.Session()
    label = f"[{log_name}] " if log_name else ""

    response: Optional[requests.Response] = None
    attempt = 0
    while True:
        attempt += 1
        try:
            response = client.get(url, **kwargs)
            break
        except requests.RequestException as exc:
            final_url = _resolve_request_url(exc, url)
            print(f"[WARN] {label}Request failed for {final_url} (attempt {attempt}/{retries}): {exc}")
            if attempt >= retries:
                # Cache the failure (None) so future calls skip retrying the same dead URL.
                if use_cache:
                    with _CACHE_LOCK:
                        _GET_CACHE[url] = None
                return None
            time.sleep(retry_delay)

    # Store the successful response in the cache for subsequent callers.
    if use_cache:
        with _CACHE_LOCK:
            _GET_CACHE[url] = response

    # Caller-specified status codes are treated as acceptable — log and return without raising.
    if response.status_code in allowed:
        print(f"[INFO] {label}HTTP {response.status_code} for {response.url}")
        return response

    if raise_for_status:
        try:
            response.raise_for_status()
        except requests.RequestException as exc:
            final_url = _resolve_request_url(exc, response.url)
            print(f"[WARN] {label}Request failed for {final_url}: {exc}")
            return None

    return response


def _resolve_request_url(exc: requests.RequestException, fallback: str) -> str:
    # Extract the request URL from the exception object, falling back to the provided default.
    request = getattr(exc, "request", None)
    if request is not None:
        return getattr(request, "url", fallback)
    return fallback


def set_thread_session(session: requests.Session | None) -> None:
    # Store or clear a requests.Session on the thread-local object for reuse by http_get.
    if session is None:
        if hasattr(THREAD_SESSION, "session"):
            del THREAD_SESSION.session
    else:
        THREAD_SESSION.session = session

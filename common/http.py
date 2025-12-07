from typing import Iterable, Optional
import time
import threading

import requests

THREAD_SESSION = threading.local()


def http_get(
    url: str,
    *,
    session: Optional[requests.Session] = None,
    log_name: Optional[str] = None,
    allowed_statuses: Optional[Iterable[int]] = None,
    raise_for_status: bool = True,
    retries: int = 3,
    retry_delay: float = 1.0,
    **kwargs,
):
    """
    Issue an HTTP GET request with consistent logging.

    Returns the Response on success or None if an exception occurred.
    """
    allowed = set(allowed_statuses or [])
    client = session or getattr(THREAD_SESSION, "session", None) or requests.Session()
    label = f"[{log_name}] " if log_name else ""

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
                return None
            time.sleep(retry_delay)

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
    request = getattr(exc, "request", None)
    if request is not None:
        return getattr(request, "url", fallback)
    return fallback


def set_thread_session(session: requests.Session | None) -> None:
    if session is None:
        if hasattr(THREAD_SESSION, "session"):
            del THREAD_SESSION.session
    else:
        THREAD_SESSION.session = session

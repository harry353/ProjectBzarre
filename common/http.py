from typing import Iterable, Optional

import requests


def http_get(
    url: str,
    *,
    session: Optional[requests.Session] = None,
    log_name: Optional[str] = None,
    allowed_statuses: Optional[Iterable[int]] = None,
    raise_for_status: bool = True,
    **kwargs,
):
    """
    Issue an HTTP GET request with consistent logging.

    Returns the Response on success or None if an exception occurred.
    """
    allowed = set(allowed_statuses or [])
    client = session or requests.Session()
    label = f"[{log_name}] " if log_name else ""

    try:
        response = client.get(url, **kwargs)
    except requests.RequestException as exc:
        final_url = _resolve_request_url(exc, url)
        print(f"[WARN] {label}Request failed for {final_url}: {exc}")
        return None

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

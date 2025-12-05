from datetime import datetime, date, timedelta
import time


def format_date(value):
    """
    Format datetime/date objects as 'YYYY-Mon-DD'.
    """
    if isinstance(value, datetime):
        value = value.date()
    if isinstance(value, date):
        return value.strftime("%Y-%b-%d")
    return str(value)


class SpaceWeatherAPI:
    DEFAULT_DOWNLOAD_RETRIES = 3
    RETRY_SLEEP_SECONDS = 5
    """
    Base class for all space weather data sources.

    The `days` argument may take several forms:

        1. Integer n
           Fetch data from (today - n + 1) through today.

        2. Single date object
           Fetch data from that date through today.

        3. Tuple of (date, date)
           Explicit start and end dates.

        4. Tuple of (date, timedelta)
           Start at the given date and extend for the given duration.

    Subclasses must implement:
        download()
        ingest(df, ...)
        plot(df)
    """

    def __init__(self, days):
        self.start_date, self.end_date = self._parse_days_argument(days)

    # ------------------------------------------------------------
    # days argument parser
    # ------------------------------------------------------------
    def _parse_days_argument(self, days):
        """
        Convert the `days` argument into (start_date, end_date).

        Returns
        -------
        (datetime.date, datetime.date)
        """

        # --------------------------------------------------------
        # Case 1: days is an integer
        # --------------------------------------------------------
        if isinstance(days, int):
            if days <= 0:
                raise ValueError("Integer days must be positive.")
            end = date.today()
            start = end - timedelta(days=days - 1)
            return start, end

        # --------------------------------------------------------
        # Case 2: days is a single date object
        # --------------------------------------------------------
        if isinstance(days, date):
            start = days
            end = date.today()
            if start > end:
                raise ValueError("Start date cannot be after today.")
            return start, end

        # --------------------------------------------------------
        # Case 3 and 4: tuple-based cases
        # --------------------------------------------------------
        if isinstance(days, tuple):
            if len(days) != 2:
                raise ValueError("Tuple days argument must have length 2.")

            a, b = days

            # Case 3: (date, date)
            if isinstance(a, date) and isinstance(b, date):
                if a > b:
                    raise ValueError("Start date cannot be after end date.")
                return a, b

            # Case 4: (date, timedelta)
            if isinstance(a, date) and isinstance(b, timedelta):
                start = a
                end = a + b
                if start > end:
                    raise ValueError("Start date cannot be after computed end date.")
                return start, end

            raise ValueError(
                "Tuple days argument must be either (date, date) or (date, timedelta)."
            )

        # --------------------------------------------------------
        # Anything else is invalid
        # --------------------------------------------------------
        raise TypeError(
            "days must be an integer, a date, a tuple of dates, or a tuple of (date, timedelta)."
        )

    # ------------------------------------------------------------
    # Abstract methods
    # ------------------------------------------------------------
    def download(self):
        """
        Execute the subclass-provided download implementation with retries.
        """
        attempts = max(1, getattr(self, "DEFAULT_DOWNLOAD_RETRIES", 1))
        delay = max(0, getattr(self, "RETRY_SLEEP_SECONDS", 0))
        last_exc = None

        for attempt in range(1, attempts + 1):
            try:
                return self._download_impl()
            except Exception as exc:
                last_exc = exc
                if attempt == attempts:
                    break
                cls_name = self.__class__.__name__
                print(
                    f"[WARN] {cls_name} download attempt {attempt} failed: {exc}. "
                    f"Retrying in {delay} seconds..."
                )
                if delay:
                    time.sleep(delay)

        if last_exc:
            raise last_exc
        raise RuntimeError("Download failed without exception.")

    def _download_impl(self):
        raise NotImplementedError("Subclasses must implement _download_impl().")

    def ingest(self, df, warehouse=None, db_path="space_weather.db"):
        raise NotImplementedError("Subclasses must implement ingest().")

    def plot(self, df):
        raise NotImplementedError("Subclasses must implement plot().")

    # ------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------
    def iter_days(self):
        """
        Yield every day from start_date to end_date inclusive.
        """
        current = self.start_date
        while current <= self.end_date:
            yield current
            current += timedelta(days=1)

    def range_str(self):
        """
        Human readable date range for logging or debugging.
        """
        return f"{format_date(self.start_date)} -> {format_date(self.end_date)}"

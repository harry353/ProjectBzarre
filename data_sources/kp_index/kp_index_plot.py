import matplotlib.pyplot as plt
import pandas as pd

# Centered smoothing window length, in days, applied after daily averaging.
SMOOTHING_WINDOW_DAYS = 365  # tweak to adjust the running-mean horizon


def plot_kp_index(df: pd.DataFrame):
    """
    Plot GFZ Kp index readings using daily averages and a centered running mean.

    Pipeline:
        1) Convert raw (3-hour cadence) Kp measurements to daily means.
        2) Apply a SMOOTHING_WINDOW_DAYS-day centered rolling average (NOAA style).
    """
    if df.empty:
        raise ValueError("Cannot plot an empty Kp data frame.")

    payload = df.copy()
    payload["time_tag"] = pd.to_datetime(payload["time_tag"], utc=True, errors="coerce")
    payload = payload.dropna(subset=["time_tag", "kp_index"]).sort_values("time_tag")
    payload = payload.set_index("time_tag")

    daily = payload["kp_index"].resample("1D").mean()
    if daily.empty:
        raise ValueError("Daily resampling produced no Kp values to plot.")

    smoothed = daily.rolling(
        window=SMOOTHING_WINDOW_DAYS, center=True, min_periods=1
    ).mean()
    smoothed = smoothed.dropna()
    if smoothed.empty:
        raise ValueError("Smoothing removed all Kp values; consider expanding the range.")

    plt.figure(figsize=(12, 6))
    plt.plot(smoothed.index, smoothed.values, linewidth=1.8, color="#d62728")

    start = smoothed.index[0]
    end = smoothed.index[-1]
    plt.title(
        "GFZ Kp Index (daily mean + centered running mean)"
        f"\n{start.date()} → {end.date()} | window = {SMOOTHING_WINDOW_DAYS} days"
    )
    plt.xlabel("Date")
    plt.ylabel("Kp (smoothed)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    ymin = float(smoothed.min())
    ymax = float(smoothed.max())
    if pd.notna(ymin) and pd.notna(ymax):
        padding = max((ymax - ymin) * 0.2, 0.2)
        plt.ylim(max(0, ymin - padding), ymax + padding)

    plt.show()


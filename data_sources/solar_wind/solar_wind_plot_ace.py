from __future__ import annotations

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


def plot_solar_wind_ace(df: pd.DataFrame) -> None:
    if df.empty:
        print("[WARN] ACE dataframe empty. Cannot plot.")
        return

    frame = df.copy()
    idx = pd.to_datetime(frame["time_tag"], errors="coerce", utc=True)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_convert(None)
    frame = frame.assign(time_tag=idx).dropna(subset=["time_tag"]).set_index("time_tag")
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    speed = frame["speed"].resample("1h").mean().dropna()
    density = frame["density"].resample("1h").mean().dropna()
    temperature = frame["temperature"].resample("1h").mean().dropna()

    axes[0].plot(speed.index, speed.values)
    axes[0].set_ylabel("Speed (km/s)")
    axes[0].set_title("ACE SWEPAM – Speed")

    axes[1].plot(density.index, density.values, color="tab:orange")
    axes[1].set_ylabel("Density (cm^-3)")
    axes[1].set_title("ACE SWEPAM – Density")

    axes[2].plot(temperature.index, temperature.values, color="tab:red")
    axes[2].set_ylabel("Temperature (K)")
    axes[2].set_title("ACE SWEPAM – Temperature")

    date_formatter = mdates.DateFormatter("%Y-%b-%d")
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(date_formatter)

    plt.tight_layout()
    fig.autofmt_xdate()
    plt.show()

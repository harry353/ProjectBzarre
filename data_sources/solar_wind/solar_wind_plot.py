import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd


def plot_solar_wind(df: pd.DataFrame):
    """
    Plot hourly averages for density, speed, and temperature.
    """
    if df.empty:
        print("[WARN] Plasma dataframe empty. Cannot plot.")
        return

    df = df.set_index("time_tag")

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    df["speed"].resample("1h").mean().plot(ax=axes[0])
    axes[0].set_ylabel("Speed (km/s)")
    axes[0].set_title("Proton Speed")

    df["density"].resample("1h").mean().plot(ax=axes[1])
    axes[1].set_ylabel("Density (cm^-3)")
    axes[1].set_title("Proton Density")

    df["temperature"].resample("1h").mean().plot(ax=axes[2])
    axes[2].set_ylabel("Temperature (K)")
    axes[2].set_title("Proton Temperature")

    date_formatter = mdates.DateFormatter("%Y-%b-%d")
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(date_formatter)

    plt.tight_layout()
    fig.autofmt_xdate()
    plt.show()

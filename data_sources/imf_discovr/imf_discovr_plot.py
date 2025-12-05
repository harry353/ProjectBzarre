import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd


def plot_imf_discovr(df: pd.DataFrame):
    """
    Visualize IMF components with hourly averages.
    """
    if df.empty:
        print("[WARN] IMF dataframe empty. Cannot plot.")
        return

    df = df.set_index("time_tag")

    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

    df["bt"].resample("1h").mean().plot(ax=axes[0])
    axes[0].set_ylabel("|B| (nT)")
    axes[0].set_title("IMF Magnitude")

    df["bx"].resample("1h").mean().plot(ax=axes[1])
    axes[1].set_ylabel("Bx GSE (nT)")

    df["by"].resample("1h").mean().plot(ax=axes[2])
    axes[2].set_ylabel("By GSE (nT)")

    df["bz"].resample("1h").mean().plot(ax=axes[3])
    axes[3].set_ylabel("Bz GSE (nT)")

    date_formatter = mdates.DateFormatter("%Y-%b-%d")
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(date_formatter)

    plt.tight_layout()
    fig.autofmt_xdate()
    plt.show()

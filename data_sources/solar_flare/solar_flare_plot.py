import matplotlib.pyplot as plt
import pandas as pd

COLOR_MAP = {"A": "gray", "B": "blue", "C": "green", "M": "orange", "X": "red"}


def plot_flares(df: pd.DataFrame, title_suffix: str = ""):
    """
    Plot flare frequency as daily bar counts colored by class.
    """
    if df.empty:
        raise ValueError("No solar flare data available to plot.")

    counts = (
        df.groupby([df["beginTime"].dt.date, "classLetter"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=sorted(COLOR_MAP.keys()))
    )

    full_range = pd.date_range(
        df["beginTime"].dt.floor("D").min(),
        df["beginTime"].dt.floor("D").max(),
        freq="D",
    )
    counts = counts.reindex(full_range.date, fill_value=0)
    counts.index = [d.strftime("%Y-%b-%d") for d in counts.index]

    if counts.empty:
        raise ValueError("No solar flare data available to plot.")

    counts.plot(
        kind="bar",
        stacked=True,
        figsize=(12, 6),
        color=[COLOR_MAP[key] for key in counts.columns],
    )

    plt.xlabel("Date")
    plt.ylabel("Number of Flares")
    suffix = f" ({title_suffix})" if title_suffix else ""
    plt.title(f"Solar Flare Counts by Class{suffix}")
    plt.legend(title="Flare Class")
    plt.tight_layout()
    plt.show()

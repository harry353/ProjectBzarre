import matplotlib.pyplot as plt
import pandas as pd


def plot_cme(df: pd.DataFrame):
    """
    Visualize CME counts per day using the DONKI CMEAnalysis dataset.
    """
    if df.empty:
        raise ValueError("No CME data available to plot.")

    counts = df.groupby(df["time21_5"].dt.date).size()
    counts.index = [d.strftime("%Y-%b-%d") for d in counts.index]

    plt.figure(figsize=(10, 4))
    counts.plot(kind="bar")
    plt.title("CME Count per Day (CMEAnalysis)")
    plt.xlabel("Date")
    plt.ylabel("Number of CMEs")
    plt.tight_layout()
    plt.show()

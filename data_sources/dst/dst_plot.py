import matplotlib.pyplot as plt
import pandas as pd


def plot_dst(df: pd.DataFrame):
    """
    Plot the Dst index time series.
    """
    if df.empty:
        raise ValueError("Cannot plot an empty Dst dataset.")

    payload = df.copy()
    payload["time_tag"] = pd.to_datetime(payload["time_tag"], errors="coerce")
    payload = payload.dropna(subset=["time_tag"]).sort_values("time_tag")

    plt.figure(figsize=(14, 5))
    plt.plot(payload["time_tag"], payload["dst"], linewidth=1.0, color="#1f77b4")
    plt.title("Kyoto WDC Dst Index")
    plt.xlabel("Date")
    plt.ylabel("Dst [nT]")
    plt.grid(True, alpha=0.3)
    # plt.ylim(-500, 100)
    plt.tight_layout()
    plt.show()


import matplotlib.pyplot as plt
import pandas as pd


def plot_sunspot_numbers(df: pd.DataFrame):
    """
    Plot the daily sunspot number series for the provided data frame.
    """
    if df.empty:
        raise ValueError("Cannot plot an empty sunspot number data frame.")

    payload = df.sort_values("time_tag").reset_index(drop=True)

    plt.figure(figsize=(12, 6))
    plt.plot(payload["time_tag"], payload["sunspot_number"], linewidth=1.5, color="#1f77b4")

    start = payload["time_tag"].iloc[0]
    end = payload["time_tag"].iloc[-1]
    plt.title(f"GFZ Sunspot Number Index ({start.date()} → {end.date()})")
    plt.xlabel("Date")
    plt.ylabel("Sunspot Number (SN)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    ymin = payload["sunspot_number"].min()
    ymax = payload["sunspot_number"].max()
    if pd.notna(ymin) and pd.notna(ymax) and ymin != ymax:
        padding = max((ymax - ymin) * 0.1, 5)
        plt.ylim(ymin - padding, ymax + padding)

    plt.show()


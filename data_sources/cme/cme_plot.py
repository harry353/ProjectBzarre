import matplotlib.pyplot as plt
import pandas as pd


def plot_cme_velocity(df: pd.DataFrame):
    """
    Plot CACTUS CME median velocity against onset time.
    """
    if df.empty:
        raise ValueError("Cannot plot an empty CME dataset.")

    payload = df.copy()
    payload["time_tag"] = pd.to_datetime(payload["time_tag"], errors="coerce")
    payload["median_velocity"] = pd.to_numeric(payload["median_velocity"], errors="coerce")
    payload = payload.dropna(subset=["time_tag", "median_velocity"])
    payload = payload.sort_values("time_tag")

    if payload.empty:
        raise ValueError("CME dataframe has no valid rows for plotting.")

    plt.figure(figsize=(10, 5))
    plt.plot(payload["time_tag"], payload["median_velocity"], marker="o", linestyle="-", color="#1f77b4")
    plt.title("CACTUS CME Median Velocity")
    plt.xlabel("Onset time (UTC)")
    plt.ylabel("Median velocity (km/s)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

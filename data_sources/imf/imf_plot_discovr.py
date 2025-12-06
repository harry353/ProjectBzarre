import matplotlib.pyplot as plt
import pandas as pd


def plot_imf_discovr(df: pd.DataFrame):
    """
    Visualize DSCOVR IMF components using the shared layout.
    """
    if df.empty:
        raise ValueError("Cannot plot an empty IMF dataset.")

    payload = df.copy()
    payload["time_tag"] = pd.to_datetime(payload["time_tag"], errors="coerce")
    payload = payload.dropna(subset=["time_tag"]).sort_values("time_tag")

    for column in ("bx", "by", "bz", "bt"):
        if column not in payload.columns:
            payload[column] = None

    fig, axes = plt.subplots(4, 1, figsize=(13, 10), sharex=True)
    axes[0].plot(payload["time_tag"], payload["bt"], linewidth=0.8, color="#455a64")
    axes[0].set_ylabel("|B| (nT)")
    axes[0].set_title("DSCOVR IMF Components")
    axes[0].grid(alpha=0.3)

    axes[1].plot(payload["time_tag"], payload["bx"], linewidth=0.8, color="#0277bd")
    axes[1].set_ylabel("Bx (nT)")
    axes[1].grid(alpha=0.3)

    axes[2].plot(payload["time_tag"], payload["by"], linewidth=0.8, color="#ef6c00")
    axes[2].set_ylabel("By (nT)")
    axes[2].grid(alpha=0.3)

    axes[3].plot(payload["time_tag"], payload["bz"], linewidth=0.8, color="#2e7d32")
    axes[3].set_ylabel("Bz (nT)")
    axes[3].set_xlabel("Time (UTC)")
    axes[3].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

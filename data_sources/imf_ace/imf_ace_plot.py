import matplotlib.pyplot as plt
import pandas as pd


def plot_imf_ace(df: pd.DataFrame):
    """
    Replicate the legacy ACE magnetic field plot with four stacked panels.
    """
    if df.empty:
        raise ValueError("Cannot plot an empty IMF ACE dataset.")

    payload = df.copy()
    payload["time_tag"] = pd.to_datetime(payload["time_tag"], errors="coerce")
    payload = payload.dropna(subset=["time_tag"])

    times = payload["time_tag"].to_numpy()

    plt.figure(figsize=(12, 12))

    plt.subplot(4, 1, 1)
    plt.plot(times, payload["bx_gse"], linewidth=0.7)
    plt.ylabel("BX GSE (nT)")
    plt.title("ACE MFI Magnetic Field Components")
    plt.grid(alpha=0.3)

    plt.subplot(4, 1, 2)
    plt.plot(times, payload["by_gse"], linewidth=0.7)
    plt.ylabel("BY GSE (nT)")
    plt.grid(alpha=0.3)

    plt.subplot(4, 1, 3)
    plt.plot(times, payload["bz_gse"], linewidth=0.7)
    plt.ylabel("BZ GSE (nT)")
    plt.grid(alpha=0.3)

    plt.subplot(4, 1, 4)
    plt.plot(times, payload["bt"], linewidth=0.7)
    plt.ylabel("|B| (nT)")
    plt.xlabel("Time (UTC)")
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

import matplotlib.pyplot as plt
import pandas as pd


def plot_sw_comp(df: pd.DataFrame):
    """
    Plot ACE/SWICS composition ratios similarly to the legacy imf_ace_temp script.
    """
    if df.empty:
        raise ValueError("Cannot plot an empty SW composition dataset.")

    payload = df.copy()
    payload["time_tag"] = pd.to_datetime(payload["time_tag"], errors="coerce")
    payload = payload.dropna(subset=["time_tag"])

    times = payload["time_tag"].to_numpy()

    plt.figure(figsize=(12, 10))

    plt.subplot(4, 1, 1)
    plt.plot(times, payload["o7_o6"], marker="o")
    plt.title("ACE SWICS Composition Ratios")
    plt.ylabel("O7/O6")
    plt.grid(alpha=0.3)

    plt.subplot(4, 1, 2)
    plt.plot(times, payload["c6_c5"], marker="o")
    plt.ylabel("C6/C5")
    plt.grid(alpha=0.3)

    plt.subplot(4, 1, 3)
    plt.plot(times, payload["avg_fe_charge"], marker="o")
    plt.ylabel("Avg Fe Charge")
    plt.grid(alpha=0.3)

    plt.subplot(4, 1, 4)
    plt.plot(times, payload["fe_to_o"], marker="o")
    plt.ylabel("Fe/O")
    plt.xlabel("Time (UTC)")
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

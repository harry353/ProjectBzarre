import matplotlib.pyplot as plt
import pandas as pd


def plot_ae(df: pd.DataFrame):
    """
    Visualize AL/AU envelopes and derived AE/AO values over time.
    """
    if df.empty:
        raise ValueError("Cannot plot an empty AE dataset.")

    payload = df.copy()
    payload["time_tag"] = pd.to_datetime(payload["time_tag"], errors="coerce")
    payload = payload.dropna(subset=["time_tag"]).sort_values("time_tag")
    payload = payload.set_index("time_tag")

    # Compute missing derivatives in case they were dropped upstream.
    if "ae" not in payload or payload["ae"].isna().all():
        payload["ae"] = payload["au"] - payload["al"]
    if "ao" not in payload or payload["ao"].isna().all():
        payload["ao"] = (payload["au"] + payload["al"]) / 2

    fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

    ax = axes[0]
    ax.plot(payload.index, payload["al"], color="tab:blue", linewidth=1.2)
    ax.plot(payload.index, payload["au"], color="tab:blue", linewidth=1.2)
    ax.fill_between(
        payload.index,
        payload["al"],
        payload["au"],
        color="tab:blue",
        alpha=0.25,
    )
    ax.set_ylabel("AL / AU (nT)")
    ax.set_title("Realtime AL and AU")
    ax.set_ylim(-2000, 1000)
    ax.grid(alpha=0.3)
    ax.legend(["Upper envelope: AU", "Lower envelope: AL"], loc="upper right")

    ax = axes[1]
    ax.plot(payload.index, payload["ae"], color="tab:red", linewidth=1.2)
    ax.plot(payload.index, payload["ao"], color="tab:red", linewidth=1.2)
    ax.fill_between(
        payload.index,
        payload["ae"],
        payload["ao"],
        color="tab:red",
        alpha=0.25,
    )
    ax.set_ylabel("AE / AO (nT)")
    ax.set_title("Derived AE and AO")
    ax.set_ylim(-500, 2000)
    ax.grid(alpha=0.3)
    ax.legend(["Upper envelope: AE", "Lower envelope: AO"], loc="upper right")

    plt.tight_layout()
    plt.show()


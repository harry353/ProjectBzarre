import matplotlib.pyplot as plt
import pandas as pd


def plot_supermag(df: pd.DataFrame):
    """
    Plot SML/SMU envelopes and derived SME/SMO series.
    """
    if df.empty:
        raise ValueError("Cannot plot an empty SuperMAG dataset.")

    payload = df.copy()

    if "time" in payload:
        payload["time"] = pd.to_datetime(payload["time"], errors="coerce")
    elif "time_tag" in payload:
        payload["time"] = pd.to_datetime(payload["time_tag"], errors="coerce")
    elif "tval" in payload:
        payload["time"] = pd.to_datetime(payload["tval"], unit="s", errors="coerce")
    else:
        payload["time"] = pd.NaT

    payload = payload.dropna(subset=["time"]).sort_values("time")
    payload = payload.set_index("time")

    if "SME" not in payload or payload["SME"].isna().all():
        if "SMU" not in payload or "SML" not in payload:
            raise ValueError("SuperMAG dataframe missing required SMU/SML columns.")
        payload["SME"] = payload["SMU"] - payload["SML"]

    if "SMO" not in payload or payload["SMO"].isna().all():
        if "SMU" not in payload or "SML" not in payload:
            raise ValueError("SuperMAG dataframe missing required SMU/SML columns.")
        payload["SMO"] = (payload["SMU"] + payload["SML"]) / 2

    fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

    ax = axes[0]
    ax.plot(payload.index, payload["SML"], color="tab:blue", linewidth=1.2)
    ax.plot(payload.index, payload["SMU"], color="tab:blue", linewidth=1.2)
    ax.fill_between(
        payload.index,
        payload["SML"],
        payload["SMU"],
        color="tab:blue",
        alpha=0.25,
    )
    ax.set_ylabel("SML / SMU (nT)")
    ax.set_title("SuperMAG SML and SMU")
    ax.grid(alpha=0.3)
    ax.legend(["Upper envelope: SMU", "Lower envelope: SML"], loc="upper right")

    ax = axes[1]
    ax.plot(payload.index, payload["SME"], color="tab:red", linewidth=1.2)
    ax.plot(payload.index, payload["SMO"], color="tab:red", linewidth=1.2)
    ax.fill_between(
        payload.index,
        payload["SME"],
        payload["SMO"],
        color="tab:red",
        alpha=0.25,
    )
    ax.set_ylabel("SME / SMO (nT)")
    ax.set_title("Derived SME and SMO")
    ax.grid(alpha=0.3)
    ax.legend(["Upper envelope: SME", "Lower envelope: SMO"], loc="upper right")

    plt.tight_layout()
    plt.show()

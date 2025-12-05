import matplotlib.pyplot as plt
import pandas as pd


def plot_radio_flux(df: pd.DataFrame):
    """
    Plot observed, adjusted, and URSI radio flux series.
    """
    if df.empty:
        raise ValueError("Cannot plot an empty radio flux data frame.")

    payload = df.copy()
    payload["time_tag"] = pd.to_datetime(payload["time_tag"], errors="coerce")
    payload = payload.dropna(subset=["time_tag"]).sort_values("time_tag")

    plt.figure(figsize=(12, 5))
    plt.scatter(payload["time_tag"], payload["observed_flux"], label="Observed (F10.7)", s=12)
    plt.scatter(payload["time_tag"], payload["adjusted_flux"], label="Adjusted (1 AU)", s=12)
    plt.scatter(payload["time_tag"], payload["ursi_flux"], label="URSI (Series D)", s=12)

    plt.title("Penticton F10.7 Solar Radio Flux")
    plt.xlabel("Date")
    plt.ylabel("Flux (sfu)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


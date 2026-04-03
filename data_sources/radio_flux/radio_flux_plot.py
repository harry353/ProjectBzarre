import matplotlib.pyplot as plt
import pandas as pd


def plot_radio_flux(df: pd.DataFrame):
    """
    Plot adjusted radio flux series.
    """
    if df.empty:
        raise ValueError("Cannot plot an empty radio flux data frame.")

    payload = df.copy()
    # Re-parse time_tag in case it was stored as strings (e.g. after loading from SQLite)
    payload["time_tag"] = pd.to_datetime(payload["time_tag"], errors="coerce")
    # Drop rows that cannot be placed on the time axis or have no adjusted flux to anchor the view
    payload = payload.dropna(subset=["time_tag", "adjusted_flux"]).sort_values("time_tag")

    # Wide figure to give the time axis enough room for multi-year date ranges
    plt.figure(figsize=(12, 5))
    # Three distinct series shown as scatter points so individual daily readings remain visible
    plt.scatter(payload["time_tag"], payload["observed_flux"], label="Observed (F10.7)", s=12)
    # Adjusted flux normalises measurements to the standard 1 AU Earth–Sun distance
    plt.scatter(payload["time_tag"], payload["adjusted_flux"], label="Adjusted (1 AU)", s=12)
    # URSI Series D is an independently derived flux index for cross-reference
    plt.scatter(payload["time_tag"], payload["ursi_flux"], label="URSI (Series D)", s=12)

    plt.title("Penticton F10.7 Solar Radio Flux")
    plt.xlabel("Date")
    # Solar flux unit (sfu) = 10^-22 W m^-2 Hz^-1
    plt.ylabel("Flux (sfu)")
    # Light grid aids reading off values without cluttering the scatter points
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

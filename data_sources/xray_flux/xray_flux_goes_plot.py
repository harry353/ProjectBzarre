import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd


def compute_primary_irradiance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute NOAA-style primary irradiances for XRS-A (short band)
    and XRS-B (long band) using the EXIS L1b selection flags.

    This matches NOAA's operational X-ray flux product.

    Primary selection rules:
        primary_xrsa == 0 -> use irradiance_xrsa1
        primary_xrsa == 1 -> use irradiance_xrsa2
        primary_xrsb == 0 -> use irradiance_xrsb1
        primary_xrsb == 1 -> use irradiance_xrsb2
    """

    # Ensure needed variables exist
    required_cols = [
        "irradiance_xrsa1", "irradiance_xrsa2",
        "irradiance_xrsb1", "irradiance_xrsb2",
        "primary_xrsa", "primary_xrsb"
    ]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required XRS L1b column(s): {', '.join(missing)}")

    df = df.copy()
    df["xrs_a"] = np.where(
        df["primary_xrsa"] == 0,
        df["irradiance_xrsa1"],
        df["irradiance_xrsa2"],
    )
    df["xrs_b"] = np.where(
        df["primary_xrsb"] == 0,
        df["irradiance_xrsb1"],
        df["irradiance_xrsb2"],
    )
    return df


def _add_xray_flare_class_grid(ax):
    """
    Add horizontal lines corresponding to X-ray flare classes:
    A, B, C, M, X (1e-8, 1e-7, 1e-6, 1e-5, 1e-4 W/m^2)
    """

    flare_levels = {
        "A": 1e-8,
        "B": 1e-7,
        "C": 1e-6,
        "M": 1e-5,
        "X": 1e-4,
    }

    for label, level in flare_levels.items():
        ax.axhline(level, color="gray", alpha=0.25, linestyle="--")
        ax.text(
            ax.get_xlim()[0],
            level,
            f" {label}",
            verticalalignment="bottom",
            color="gray",
            alpha=0.7,
        )


def plot_xrs_goes(df: pd.DataFrame):
    """
    Plot NOAA-style GOES XRS flux using primary irradiance selection.

    Parameters
    ----------
    df : XRayFluxFrame
        Frame returned by the downloader BEFORE ingestion.
    """

    if df.empty:
        print("[WARN] XRS dataframe empty. Cannot plot.")
        return

    # Compute xrs_a and xrs_b (NOAA primary irradiance)
    df = compute_primary_irradiance(df)

    fig, ax = plt.subplots(1, 1, figsize=(14, 6))

    times = df.index if df.index.name == "time_tag" else df["time_tag"]
    ax.plot(times, df["xrs_a"], label="Short (0.05–0.4 nm)")
    ax.plot(times, df["xrs_b"], label="Long (0.1–0.8 nm)")

    ax.set_yscale("log")
    ax.set_ylabel("W/m^2")
    ax.set_title("GOES XRS Primary Irradiance (NOAA-style)")

    # Add flare class gridlines
    _add_xray_flare_class_grid(ax)

    ax.legend()
    ax.grid(True, which="both", ls="--", alpha=0.3)
    date_formatter = mdates.DateFormatter("%Y-%b-%d")
    ax.xaxis.set_major_formatter(date_formatter)
    fig.autofmt_xdate()

    plt.tight_layout()
    plt.show()

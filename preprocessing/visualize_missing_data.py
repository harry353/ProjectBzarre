from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import DB_PATH

CONTINUOUS_TABLES: Dict[str, Tuple[str, List[str], str]] = {
    "ace_swepam": ("time_tag", ["density", "speed", "temperature"], "ACE Solar Wind"),
    "dscovr_f1m": ("time_tag", ["density", "speed", "temperature"], "DSCOVR Solar Wind"),
    "ace_mfi": ("time_tag", ["bx_gse", "by_gse", "bz_gse", "bt"], "ACE IMF"),
    "dscovr_m1m": ("time_tag", ["bt", "bx", "by", "bz"], "DSCOVR IMF"),
    "xray_flux": ("time_tag", ["irradiance_xrsa1", "irradiance_xrsb1"], "GOES X-Ray Flux"),
    "radio_flux": ("time_tag", ["observed_flux", "adjusted_flux", "ursi_flux"], "Radio Flux"),
    "ae_indices": ("time_tag", ["al", "au", "ae", "ao"], "AE Indices"),
    "dst_index": ("time_tag", ["dst"], "Dst Index"),
    "kp_index": ("time_tag", ["kp_index"], "Kp Index"),
    "supermag_indices": ("time_tag", ["sml", "smu", "sme", "smo"], "SuperMAG"),
    "ace_swics_composition": ("time_tag", ["o7_o6", "c6_c5", "avg_fe_charge", "fe_to_o"], "ACE Composition"),
    "sunspot_numbers": ("time_tag", ["sunspot_number"], "Sunspot Number"),
}


def load_table(name: str, time_col: str, value_cols: List[str]) -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query(
            f"SELECT {time_col}, {', '.join(value_cols)} FROM {name}",
            conn,
            parse_dates=[time_col],
        )
    if df.empty:
        return df
    df = df.rename(columns={col: f"{name}.{col}" for col in value_cols})
    df = df.set_index(time_col).sort_index()
    return df


def visualize_missingness(df: pd.DataFrame, table: str, label: str) -> None:
    if df.empty:
        print(f"[WARN] {table}: no data to visualize.")
        return
    mask = df.notna().astype(int)
    if len(mask) > 5000:
        step = int(np.ceil(len(mask) / 5000))
        mask = mask.iloc[::step]
    fig, ax = plt.subplots(figsize=(12, max(3, len(df.columns) * 0.4)))
    cax = ax.imshow(mask.T, aspect="auto", interpolation="nearest", cmap="Greys", vmin=0, vmax=1)
    ax.set_title(f"Missingness for {label}")
    friendly_cols = [
        f"{label} - {col.split('.', 1)[-1]}" if "." in col else f"{label} - {col}"
        for col in mask.columns
    ]
    ax.set_yticks(range(len(mask.columns)))
    ax.set_yticklabels(friendly_cols)
    ax.set_xlabel("Timestamp")
    xtick_positions = []
    start_year = mask.index[0].year
    end_year = mask.index[-1].year
    for year in range(start_year, end_year + 1):
        january_first = pd.Timestamp(year=year, month=1, day=1, tz=mask.index.tz)
        if january_first < mask.index[0]:
            january_first = mask.index[0]
        if january_first > mask.index[-1]:
            break
        position = mask.index.get_indexer([january_first], method="nearest")[0]
        xtick_positions.append(position)
    xtick_positions = np.unique(np.clip(xtick_positions, 0, len(mask) - 1))
    ax.set_xticks(xtick_positions)
    xtick_labels = mask.index[xtick_positions]
    ax.set_xticklabels([ts.strftime("%Y-%m") for ts in xtick_labels], rotation=45, ha="right")
    ax.set_ylabel("Columns")
    fig.colorbar(cax, ax=ax, ticks=[0, 1], label="Missing (0) / Present (1)")
    fig.tight_layout()
    out_dir = Path(__file__).resolve().parent / "figures" / "missingness"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{table}_missingness.png")
    plt.close(fig)
    print(f"[INFO] Saved missingness plot for {table}")


def main() -> None:
    if not DB_PATH.exists():
        raise FileNotFoundError(f"No database found at {DB_PATH}")
    for table, (time_col, value_cols, label) in CONTINUOUS_TABLES.items():
        df = load_table(table, time_col, value_cols)
        visualize_missingness(df, table, label)


if __name__ == "__main__":
    main()

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

STAGE_DIR = Path(__file__).resolve().parent
PIPELINE_ROOT = STAGE_DIR.parent
KP_DB = PIPELINE_ROOT / "kp_index" / "1_averaging" / "kp_index_aver.db"

KP_BINS = [
    ("0 ≤ Kp < 5", "G0", 0.0, 5.0),
    ("5 ≤ Kp < 6", "G1", 5.0, 6.0),
    ("6 ≤ Kp < 7", "G2", 6.0, 7.0),
    ("7 ≤ Kp < 8", "G3", 7.0, 8.0),
    ("8 ≤ Kp < 9", "G4", 8.0, 9.0),
    ("Kp ≥ 9", "G5", 9.0, None),
]


def _load_kp_series() -> pd.Series:
    if not KP_DB.exists():
        raise FileNotFoundError(f"Kp averaging database missing: {KP_DB}")
    with sqlite3.connect(KP_DB) as conn:
        df = pd.read_sql_query("SELECT kp_index FROM hourly_data", conn)
    if "kp_index" not in df.columns:
        raise RuntimeError("kp_index column missing from Kp hourly dataset.")
    return df["kp_index"].astype(float).dropna()


def _count_bins(kp_values: pd.Series) -> list[tuple[str, str, int]]:
    counts: list[tuple[str, str, int]] = []
    for label, gclass, lower, upper in KP_BINS:
        mask = kp_values >= lower
        if upper is not None:
            mask &= kp_values < upper
        counts.append((label, gclass, int(mask.sum())))
    return counts


def _print_report(counts: list[tuple[str, str, int]], total: int) -> None:
    label_width = max(len(label) for label, _, _ in counts)
    print("Kp Distribution (entire dataset)")
    print("-" * (label_width + 12))
    for label, gclass, count in counts:
        percentage = (count / total * 100.0) if total else 0.0
        print(f"{label:<{label_width}} ({gclass}) : {count:10,d}  ({percentage:6.3f}%)")
    print("-" * (label_width + 12))
    print(f"{'Total':<{label_width}} : {total:10,d}")

    filtered_total = sum(count for label, _, count in counts if label != "0 ≤ Kp < 5")
    if filtered_total:
        print("\nPercentages excluding Kp < 5")
        print("-" * (label_width + 12))
        for label, gclass, count in counts:
            if label == "0 ≤ Kp < 5":
                continue
            percentage = count / filtered_total * 100.0
            print(f"{label:<{label_width}} ({gclass}) : {count:10,d}  ({percentage:6.2f}%)")
        print("-" * (label_width + 12))
        print(f"{'Total>=5':<{label_width}} : {filtered_total:10,d}")


def main() -> None:
    kp_values = _load_kp_series()
    counts = _count_bins(kp_values)
    _print_report(counts, len(kp_values))


if __name__ == "__main__":
    main()

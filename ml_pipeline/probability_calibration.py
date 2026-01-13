from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = PROJECT_ROOT / "ml_pipeline" / "horizon_models"

TRAIN_TABLE = "train_probs"
VAL_TABLE = "validation_probs"
RAW_TABLE = "raw_probs"
OUT_TABLE = "calibrated_probs"
TARGET_HORIZONS_H = range(1, 9)


def _load_tables(
    raw_prob_db: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    with sqlite3.connect(raw_prob_db) as conn:
        tables = pd.read_sql_query(
            "SELECT name FROM sqlite_master WHERE type='table'", conn
        )["name"].tolist()

        if RAW_TABLE in tables:
            raw = pd.read_sql_query(f"SELECT * FROM {RAW_TABLE}", conn)
            if "split" not in raw.columns:
                raise RuntimeError("raw_probs missing required 'split' column")
            train = raw.loc[raw["split"] == "train"].copy()
            val = raw.loc[raw["split"] == "validation"].copy()
            test = raw.loc[raw["split"] == "test"].copy()
            if test.empty:
                test = None
            return train, val, test

        if TRAIN_TABLE in tables and VAL_TABLE in tables:
            train = pd.read_sql_query(f"SELECT * FROM {TRAIN_TABLE}", conn)
            val = pd.read_sql_query(f"SELECT * FROM {VAL_TABLE}", conn)
            return train, val, None

    raise RuntimeError("No raw probability tables found.")


def main() -> None:
    all_train = []
    all_val = []
    all_test = []

    per_horizon = {}

    for horizon in TARGET_HORIZONS_H:
        raw_prob_db = MODEL_ROOT / f"h{horizon}" / "raw_probabilities.db"
        train, val, test = _load_tables(raw_prob_db)

        if not {"y_true", "y_prob"}.issubset(train.columns):
            raise RuntimeError(f"h{horizon}: train missing required columns")
        if not {"y_true", "y_prob"}.issubset(val.columns):
            raise RuntimeError(f"h{horizon}: val missing required columns")

        train = train.assign(horizon=horizon)
        val = val.assign(horizon=horizon)
        if test is not None:
            test = test.assign(horizon=horizon)

        all_train.append(train)
        all_val.append(val)
        if test is not None:
            all_test.append(test)

        per_horizon[horizon] = (train, val, test)

    train_all = pd.concat(all_train, ignore_index=True)
    val_all = pd.concat(all_val, ignore_index=True)
    test_all = pd.concat(all_test, ignore_index=True) if all_test else None

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(val_all["y_prob"].to_numpy(), val_all["y_true"].to_numpy())

    print("[INFO] -------- Joint Calibration Summary --------")
    print(
        f"[INFO] Val log loss (raw)         : "
        f"{log_loss(val_all.y_true, val_all.y_prob):.4f}"
    )
    print(
        f"[INFO] Val log loss (calibrated)  : "
        f"{log_loss(val_all.y_true, iso.transform(val_all.y_prob)): .4f}"
    )
    print("[INFO] ------------------------------------------")

    for horizon, (train, val, test) in per_horizon.items():
        calib_db = MODEL_ROOT / f"h{horizon}" / "calibrated_probabilities.db"

        train["prob_calibrated"] = iso.transform(train["y_prob"].to_numpy())
        val["prob_calibrated"] = iso.transform(val["y_prob"].to_numpy())
        if test is not None:
            test["prob_calibrated"] = iso.transform(test["y_prob"].to_numpy())

        if "split" not in train.columns:
            train = train.assign(split="train")
        if "split" not in val.columns:
            val = val.assign(split="validation")

        frames = [train, val]
        if test is not None:
            if "split" not in test.columns:
                test = test.assign(split="test")
            frames.append(test)

        out = pd.concat(frames, ignore_index=True).drop(columns=["horizon"])

        calib_db.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(calib_db) as conn:
            out.to_sql(OUT_TABLE, conn, if_exists="replace", index=False)

        print(f"[OK] h{horizon}: calibrated probabilities written to {calib_db}")
        print(f"[OK] Rows: {len(out):,}")


if __name__ == "__main__":
    main()

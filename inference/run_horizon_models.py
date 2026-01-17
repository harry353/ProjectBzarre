from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from xgboost import Booster, DMatrix

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_DB = PROJECT_ROOT / "inference" / "horizon_vector.db"
MODELS_DIR = PROJECT_ROOT / "ml_pipeline" / "horizon_models"
OUTPUT_DB = PROJECT_ROOT / "inference" / "horizon_predictions.db"
CALIBRATOR_PATH = MODELS_DIR / "calibrator.json"
HOURS_AHEAD_PREDICTION = 6

TIMESTAMP_COLS = ["timestamp", "time_tag", "date"]


def _load_selected_features(h: int) -> List[str]:
    path = MODELS_DIR / f"h{h}" / "selected_features.json"
    with path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    if not isinstance(data, list):
        raise ValueError(f"h{h}: selected_features.json must contain a list")
    return [str(x) for x in data]


def _load_model(h: int) -> Booster:
    model_path = MODELS_DIR / f"h{h}" / "model.json"
    booster = Booster()
    booster.load_model(str(model_path))
    return booster


def _detect_timestamp_column(columns: List[str]) -> Optional[str]:
    for col in TIMESTAMP_COLS:
        if col in columns:
            return col
    return None


def _prepare_matrix(df: pd.DataFrame, feature_order: List[str]) -> DMatrix:
    X = df.reindex(columns=feature_order)
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.fillna(0.0)
    return DMatrix(X.to_numpy(), feature_names=feature_order)


def _load_calibrator() -> Optional[dict]:
    if not CALIBRATOR_PATH.exists():
        print(f"[WARN] Calibrator not found at {CALIBRATOR_PATH}. Using raw probabilities.")
        return None
    with CALIBRATOR_PATH.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def _apply_calibration(probs: np.ndarray, calibrator: Optional[dict]) -> np.ndarray:
    if not calibrator:
        return probs
    x = np.array(calibrator.get("x_thresholds", []), dtype=float)
    y = np.array(calibrator.get("y_thresholds", []), dtype=float)
    if x.size == 0 or y.size == 0 or x.size != y.size:
        print("[WARN] Invalid calibrator; returning raw probabilities.")
        return probs
    return np.interp(probs, x, y, left=y[0], right=y[-1])


def _predict_horizon(h: int, conn_in: sqlite3.Connection) -> Optional[pd.DataFrame]:
    table = f"h{h}_vector"
    available_tables = {
        row[0]
        for row in conn_in.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
    }
    if table not in available_tables:
        print(f"[WARN] h{h}: table {table} not found in input DB. Skipping.")
        return None

    df = pd.read_sql_query(f"SELECT * FROM {table}", conn_in)
    if df.empty:
        print(f"[WARN] h{h}: table {table} is empty. Skipping.")
        return None

    ts_col = _detect_timestamp_column(df.columns.tolist())
    timestamp = df[ts_col] if ts_col else None

    features = _load_selected_features(h)
    missing = [f for f in features if f not in df.columns]
    if missing:
        print(f"[WARN] h{h}: missing features {missing}. Skipping.")
        return None

    booster = _load_model(h)
    dmat = _prepare_matrix(df, features)
    probs = booster.predict(dmat)
    calibrated = _apply_calibration(probs, _load_calibrator())

    out = pd.DataFrame({"y_prob": probs, "y_prob_calibrated": calibrated})
    if timestamp is not None:
        out.insert(0, ts_col, timestamp)

    out = out.rename(columns={"y_prob_calibrated": f"p_h{h}"})
    keep_cols = [c for c in out.columns if c == ts_col or c == f"p_h{h}"]
    return out[keep_cols]


def main() -> None:
    if not INPUT_DB.exists():
        raise FileNotFoundError(f"Input DB not found: {INPUT_DB}")

    OUTPUT_DB.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_DB.unlink(missing_ok=True)

    frames = []
    with sqlite3.connect(INPUT_DB) as conn_in:
        for horizon in range(1, HOURS_AHEAD_PREDICTION + 1):
            try:
                frame = _predict_horizon(horizon, conn_in)
                if frame is not None and not frame.empty:
                    frames.append(frame)
            except Exception as exc:
                print(f"[ERROR] h{horizon}: {exc}")

    if not frames:
        print("[ERROR] No horizon predictions available.")
        return

    merged = frames[0]
    ts_col = next((c for c in merged.columns if c in TIMESTAMP_COLS), None)
    for frame in frames[1:]:
        if ts_col and ts_col in frame.columns:
            merged = merged.merge(frame, on=ts_col, how="inner")
        else:
            merged = pd.concat([merged.reset_index(drop=True), frame.reset_index(drop=True)], axis=1)

    prob_cols = [c for c in merged.columns if c.startswith("p_h")]
    surv = 1.0
    for col in prob_cols:
        surv *= (1.0 - merged[col].to_numpy())
    merged["p_cumulative"] = 1.0 - surv

    # Write calibrated horizon probabilities with timestamps (append/update by timestamp if present)
    with sqlite3.connect(OUTPUT_DB) as out_conn:
        if "predictions" in {
            row[0]
            for row in out_conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
        }:
            existing = pd.read_sql_query("SELECT * FROM predictions", out_conn)
            combined = pd.concat([existing, merged], ignore_index=True)
            if ts_col and ts_col in combined.columns:
                combined = combined.drop_duplicates(subset=[ts_col], keep="last")
            merged = combined
        merged.to_sql("predictions", out_conn, if_exists="replace", index=False)

    last_row = merged.iloc[-1]
    if ts_col:
        print(f"[OK] Cumulative probability at {last_row[ts_col]}: {last_row['p_cumulative']}")
    else:
        print(f"[OK] Cumulative probability: {last_row['p_cumulative']}")


if __name__ == "__main__":
    main()

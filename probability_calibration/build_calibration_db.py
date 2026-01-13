from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve()
for parent in PROJECT_ROOT.parents:
    if (parent / "space_weather_api.py").exists():
        PROJECT_ROOT = parent
        break
else:
    PROJECT_ROOT = PROJECT_ROOT.parent

PIPELINE_ROOT = PROJECT_ROOT / "preprocessing_pipeline"
MERGED_DB = PIPELINE_ROOT / "check_multicolinearity" / "all_preprocessed_sources.db"

OUTPUT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DB = OUTPUT_DIR / "validation_calibration.db"
OUTPUT_TABLE = "validation"

MODEL_DIR_BASE = PROJECT_ROOT / "modeling_pipeline"
HORIZONS = range(1, 9)

LABEL_COL_CANDIDATES = [
    "storm_labels_storm_next_24h",
    "storm_labels_storm_severity_next_8h",
]

# ---------------------------------------------------------------------
# Solar cycle minima (authoritative, UTC)
# ---------------------------------------------------------------------
SOLAR_CYCLE_MINIMA = [
    pd.Timestamp("1996-08-01T00:00:00Z"),
    pd.Timestamp("2008-12-01T00:00:00Z"),
    pd.Timestamp("2019-12-01T00:00:00Z"),
    pd.Timestamp("2031-01-01T00:00:00Z"),  # future guard
]


def _compute_solar_cycle_phase(ts: pd.Timestamp) -> float:
    """
    Compute solar cycle phase in [0, 1] from timestamp.
    Phase = (ts - cycle_start) / (cycle_end - cycle_start)
    """
    for start, end in zip(SOLAR_CYCLE_MINIMA[:-1], SOLAR_CYCLE_MINIMA[1:]):
        if start <= ts < end:
            phase = (ts - start) / (end - start)
            return float(np.clip(phase, 0.0, 1.0))
    return np.nan


# ---------------------------------------------------------------------
# Load helpers
# ---------------------------------------------------------------------
def _load_feature_order(features_path: Path) -> list[str]:
    if not features_path.exists():
        raise FileNotFoundError(f"Missing feature contract: {features_path}")
    payload = json.loads(features_path.read_text())
    order = payload.get("feature_order", [])
    if not order:
        raise RuntimeError("Feature contract missing 'feature_order'.")
    return order


def _load_model(model_path: Path) -> XGBClassifier:
    if not model_path.exists():
        raise FileNotFoundError(f"Missing trained model: {model_path}")
    model = XGBClassifier()
    model.load_model(model_path)
    return model


def _ensure_utc(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce")
    if parsed.dt.tz is None:
        return parsed.dt.tz_localize("UTC")
    return parsed.dt.tz_convert("UTC")


def _load_merged() -> pd.DataFrame:
    if not MERGED_DB.exists():
        raise FileNotFoundError(f"Merged dataset not found at {MERGED_DB}")

    frames = []
    with sqlite3.connect(MERGED_DB) as conn:
        for split in ("train", "validation", "test"):
            table = f"merged_{split}"
            df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
            if df.empty:
                continue
            if "timestamp" not in df.columns:
                raise RuntimeError(f"Missing timestamp column in {table}")
            df["timestamp"] = _ensure_utc(df["timestamp"])
            df = df.dropna(subset=["timestamp"])
            frames.append(df)

    if not frames:
        raise RuntimeError("No merged data available.")

    merged = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates("timestamp")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    return merged


def _select_label_column(df: pd.DataFrame, horizon: int) -> str:
    desired = f"full_storm_labels_storm_flag_h{horizon}"
    if desired in df.columns:
        return desired
    for name in LABEL_COL_CANDIDATES:
        if name in df.columns:
            return name
    for col in df.columns:
        if col.startswith("full_storm_labels_") or col.startswith("storm_labels_"):
            return col
    raise RuntimeError(f"No label column found in merged dataset for h{horizon}.")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main() -> None:
    merged = _load_merged()

    calibration = pd.DataFrame(
        {
            "timestamp": merged["timestamp"],
            "solar_cycle_phase": merged["timestamp"].apply(_compute_solar_cycle_phase),
        }
    )

    for horizon in HORIZONS:
        output_dir = MODEL_DIR_BASE / f"output_h{horizon}"
        features_path = output_dir / "daily_storm_features.json"
        model_path = output_dir / "daily_storm_model.json"
        feature_order = _load_feature_order(features_path)
        model = _load_model(model_path)

        label_col = _select_label_column(merged, horizon)
        missing = [c for c in feature_order if c not in merged.columns]
        if missing:
            raise RuntimeError(f"Missing feature columns for h{horizon}: {missing}")

        X = merged[feature_order].fillna(0.0).to_numpy(dtype=np.float32)
        if not hasattr(model, "predict_proba"):
            raise RuntimeError("Model does not support predict_proba.")
        raw_prob = model.predict_proba(X)[:, 1]
        raw_prob = np.clip(raw_prob.astype(float), 0.0, 1.0)

        labels = pd.to_numeric(merged[label_col], errors="coerce")
        labels = (labels > 0).astype(int)

        calibration[f"raw_prob_h{horizon}"] = raw_prob
        calibration[f"label_h{horizon}"] = labels

    required_cols = ["timestamp", "solar_cycle_phase"]
    for horizon in HORIZONS:
        required_cols.append(f"raw_prob_h{horizon}")
        required_cols.append(f"label_h{horizon}")
    calibration = calibration.dropna(subset=required_cols)

    if calibration.empty:
        raise RuntimeError("No calibration rows after cleaning.")

    with sqlite3.connect(OUTPUT_DB) as conn:
        calibration.to_sql(OUTPUT_TABLE, conn, if_exists="replace", index=False)

    print(f"[OK] Calibration rows written: {len(calibration):,}")
    print(
        f"[OK] Date range: "
        f"{calibration['timestamp'].min()} -> {calibration['timestamp'].max()}"
    )
    print(f"[OK] Output DB: {OUTPUT_DB}")


if __name__ == "__main__":
    main()

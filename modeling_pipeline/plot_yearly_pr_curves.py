from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modeling_pipeline_daily.data_utils import TARGET_COL, load_split_tables


PIPELINE_ROOT = PROJECT_ROOT / "preprocessing_pipeline"
MERGED_DB = PIPELINE_ROOT / "check_multicolinearity" / "all_preprocessed_sources.db"

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
PLOT_DIR = OUTPUT_DIR / "yearly_pr_curves"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

FEATURES_JSON = OUTPUT_DIR / "daily_storm_features.json"
MODEL_JSON = OUTPUT_DIR / "daily_storm_model.json"


def _load_feature_order() -> list[str]:
    if not FEATURES_JSON.exists():
        raise FileNotFoundError(f"Missing feature contract: {FEATURES_JSON}")
    payload = json.loads(FEATURES_JSON.read_text())
    feature_order = payload.get("feature_order", [])
    if not feature_order:
        raise RuntimeError("Feature contract missing 'feature_order'.")
    return feature_order


def _load_model() -> XGBClassifier:
    if not MODEL_JSON.exists():
        raise FileNotFoundError(f"Missing model: {MODEL_JSON}")
    model = XGBClassifier()
    model.load_model(MODEL_JSON)
    return model


def _predict_probs(df: pd.DataFrame, feature_order: list[str], model: XGBClassifier) -> pd.Series:
    missing = [c for c in feature_order if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required feature columns: {missing}")
    X = df[feature_order].fillna(0.0).to_numpy(dtype=np.float32)
    prob = model.predict_proba(X)[:, 1]
    return pd.Series(prob, index=df["forecast_date"], name="prob")


def _plot_year(
    year: int,
    train: pd.DataFrame,
    val: pd.DataFrame,
    train_prob: pd.Series,
    val_prob: pd.Series,
) -> None:
    train_mask = (train["forecast_date"].dt.year == year).to_numpy()
    val_mask = (val["forecast_date"].dt.year == year).to_numpy()

    if not train_mask.any() and not val_mask.any():
        return

    plt.figure(figsize=(6, 4))
    plotted = False

    if train_mask.any():
        y_true = train.loc[train_mask, TARGET_COL].to_numpy(dtype=np.int8)
        y_prob = train_prob.to_numpy(dtype=float)[train_mask]
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
        plt.plot(recall, precision, label=f"Train (AP={ap:.3f})")
        plotted = True

    if val_mask.any():
        y_true = val.loc[val_mask, TARGET_COL].to_numpy(dtype=np.int8)
        y_prob = val_prob.to_numpy(dtype=float)[val_mask]
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
        plt.plot(recall, precision, label=f"Validation (AP={ap:.3f})")
        plotted = True

    if not plotted:
        plt.close()
        return

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Daily Storm PR Curves ({year})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    out_path = PLOT_DIR / f"daily_pr_{year}.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[OK] Saved {out_path}")


def main() -> None:
    feature_order = _load_feature_order()
    model = _load_model()

    splits = load_split_tables(MERGED_DB)
    train = splits["train"].copy()
    val = splits["validation"].copy()

    train_prob = _predict_probs(train, feature_order, model)
    val_prob = _predict_probs(val, feature_order, model)

    years = sorted(
        set(train["forecast_date"].dt.year).union(val["forecast_date"].dt.year)
    )
    for year in years:
        _plot_year(year, train, val, train_prob, val_prob)


if __name__ == "__main__":
    main()

from __future__ import annotations
import sys
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import brier_score_loss, log_loss, precision_recall_curve, auc
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

# --- Configuration & Paths ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURES_DB = PROJECT_ROOT / "preprocessing_pipeline" / "check_multicolinearity" / "all_preprocessed_sources.db"
LABELS_DB = PROJECT_ROOT / "preprocessing_pipeline" / "labels" / "main_phase_labels.db"
OUTPUT_ROOT = PROJECT_ROOT / "ml_pipeline" / "horizon_models"

def _load_validation_data(h: int | str):
    """Loads and merges feature/label data for a specific horizon."""
    target_table_feat = "merged_validation"
    target_table_lab = f"storm_onset_validation"
    target_col = f"h_{h}"

    print(f"[INFO] Connecting to databases for h={h}...")
    with sqlite3.connect(FEATURES_DB) as conn_f, sqlite3.connect(LABELS_DB) as conn_l:
        features = pd.read_sql_query(f"SELECT * FROM {target_table_feat}", conn_f)
        labels = pd.read_sql_query(f"SELECT * FROM {target_table_lab}", conn_l)
    
    # Ensure timestamps are clean for merging
    features["timestamp"] = pd.to_datetime(features["timestamp"]).dt.tz_localize(None)
    labels["timestamp"] = pd.to_datetime(labels["timestamp"]).dt.tz_localize(None)
    
    merged = features.merge(labels, on="timestamp", how="inner")
    numeric = merged.select_dtypes(include=[np.number]).dropna(axis=0)
    drop_cols = [c for c in numeric.columns if c.startswith("h_") and c != target_col]
    if drop_cols:
        numeric = numeric.drop(columns=drop_cols)
    
    y = numeric[target_col].astype(int).to_numpy()
    X_df = numeric.drop(columns=[target_col])
    return X_df, y

def calculate_metrics(y_true, y_prob):
    """Calculates Log Loss, Brier Score, and Brier Skill Score."""
    ll = log_loss(y_true, y_prob)
    bs = brier_score_loss(y_true, y_prob)
    
    # Baseline: what if we just guessed the average occurrence rate?
    avg_rate = y_true.mean()
    bs_ref = brier_score_loss(y_true, np.full_like(y_true, avg_rate))
    bss = 1 - (bs / bs_ref)
    
    # Area Under Precision-Recall Curve (Better for imbalanced data than ROC)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    auprc = auc(recall, precision)
    
    return ll, bs, bss, auprc

def plot_analysis(y_true, raw_prob, cal_prob, h, output_path):
    """Generates a Reliability Diagram comparing raw and calibrated outputs."""
    plt.figure(figsize=(10, 5))

    # 1. Reliability Diagram (Calibration Curve)
    plt.subplot(1, 2, 1)
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly Calibrated")
    
    # We use 'quantile' strategy because storm events are rare (< 1%)
    for p, label in zip([raw_prob, cal_prob], ["Raw XGB", "Calibrated"]):
        true_p, pred_p = calibration_curve(y_true, p, n_bins=10, strategy='quantile')
        plt.plot(pred_p, true_p, "s-", label=label)
    
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives (Actual)")
    plt.title(f"Reliability Diagram (h={h})")
    plt.legend()
    plt.grid(True)

    # 2. Precision-Recall Curve
    plt.subplot(1, 2, 2)
    precision, recall, _ = precision_recall_curve(y_true, cal_prob)
    plt.plot(recall, precision, color='blue', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'PR Curve (h={h})')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(output_path)
    print(f"[INFO] Analysis plot saved to: {output_path}")

def _load_validation_data_with_names(h: int | str):
    # This should mirror your exact training data prep
    X, y = _load_validation_data(h) 
    
    # Fetch columns from features DB to get the names in order
    with sqlite3.connect(FEATURES_DB) as conn:
        temp = pd.read_sql_query("SELECT * FROM merged_validation LIMIT 1", conn)
    
    # Re-apply your numeric/drop logic to get the final name list
    numeric = temp.select_dtypes(include=[np.number]).dropna(axis=0)
    
    # Ensure you drop the same columns you did during training!
    leaky_manual = ['kp_kp_entered_storm', 'kp_kp_ge5_flag', 'dst_dst_negative_flag', 'dst_dst_recovery_flag']
    target_col = f"h_{h}"
    drop_cols = [c for c in numeric.columns if c.startswith("h_") or c in leaky_manual]
    
    feature_names = numeric.drop(columns=drop_cols, errors='ignore').columns.tolist()
    return X, y, feature_names

def main():
    # Get horizon from CLI (default to 4)
    h = sys.argv[1] if len(sys.argv) > 1 else "4"
    h_dir = OUTPUT_ROOT / f"h{h}"
    model_path = h_dir / "model.json"
    meta_path = h_dir / "best_params.json"

    if not model_path.exists():
        print(f"[ERROR] Model file not found: {model_path}")
        print("[FIX] Run tune_hazard_xgboost.py to generate model.json")
        return

    # 1. Load Data
    X_val_df, y_val = _load_validation_data(h)
    if meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as fp:
            meta = json.load(fp)
        feature_cols = meta.get("feature_columns")
        if feature_cols:
            missing = [c for c in feature_cols if c not in X_val_df.columns]
            if missing:
                raise RuntimeError(f"Missing expected feature columns: {missing}")
            X_val_df = X_val_df[feature_cols]
    X_val = X_val_df.to_numpy(dtype=np.float32)
    
    # 2. Load Model
    model = XGBClassifier()
    model.load_model(model_path)
    
    # 3. Raw Predictions
    raw_probs = model.predict_proba(X_val)[:, 1]
    
    # 4. Calibration (Sigmoid/Platt Scaling)
    print("[INFO] Performing Post-hoc Calibration...")
    calibrator = LogisticRegression(solver="lbfgs")
    calibrator.fit(raw_probs.reshape(-1, 1), y_val)
    cal_probs = calibrator.predict_proba(raw_probs.reshape(-1, 1))[:, 1]
    
    # 5. Metrics Calculation
    metrics_raw = calculate_metrics(y_val, raw_probs)
    metrics_cal = calculate_metrics(y_val, cal_probs)
    
    # 6. Report Results
    print("\n" + "="*50)
    print(f"HAZARD MODEL EVALUATION: HORIZON h={h}")
    print("="*50)
    print(f"{'Metric':<20} | {'Raw XGB':<12} | {'Calibrated':<12}")
    print("-" * 50)
    print(f"{'Log Loss':<20} | {metrics_raw[0]:.6f}     | {metrics_cal[0]:.6f}")
    print(f"{'Brier Score':<20} | {metrics_raw[1]:.6f}     | {metrics_cal[1]:.6f}")
    print(f"{'Brier Skill Score':<20} | {metrics_raw[2]:.4f}       | {metrics_cal[2]:.4f}")
    print(f"{'AUPRC':<20} | {metrics_raw[3]:.4f}       | {metrics_cal[3]:.4f}")
    print("="*50)
    print(f"Positive samples in val: {y_val.sum()} / {len(y_val)}")

    # 7. Visualization
    plot_analysis(y_val, raw_probs, cal_probs, h, h_dir / "evaluation_plots.png")

    # --- DIAGNOSTIC: FIND THE LEAK ---
    print("\n[DIAGNOSTIC] Investigating 1.0 AUPRC...")
    
    # 1. Get the raw booster
    booster = model.get_booster()
    importance = booster.get_score(importance_type='gain')
    
    # 2. Re-run the exact X, y preparation logic used in main() to get feature names
    # We do this to ensure the list 'feature_cols' matches the 'f0, f1...' in the booster
    _, _, feature_cols = _load_validation_data_with_names(h) 

    # 3. Map internal XGBoost IDs (f0, f1...) to actual names
    mapped_importance = {}
    for k, v in importance.items():
        idx = int(k[1:]) # Convert 'f12' -> 12
        if idx < len(feature_cols):
            mapped_importance[feature_cols[idx]] = v

    sorted_map = sorted(mapped_importance.items(), key=lambda x: x[1], reverse=True)

    print("\n[TOP 10 FEATURE IMPORTANCE (BY GAIN)]")
    print("-" * 55)
    for feat, score in sorted_map[:70]:
        print(f"{feat:<35} | Gain: {score:,.2f}")
    print("-" * 55)

if __name__ == "__main__":
    main()

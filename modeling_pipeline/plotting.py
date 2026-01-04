from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve, average_precision_score


def save_pr_curve(
    y_true,
    y_prob,
    csv_path: Path,
    png_path: Path,
    average_precision: float,
) -> None:
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_df = pd.DataFrame({"recall": recall, "precision": precision})
    pr_df.to_csv(csv_path, index=False)

    plt.figure(figsize=(6, 4))
    plt.plot(recall, precision, label=f"AP={average_precision:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Daily Storm Presence Precision-Recall")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.close()


def plot_calibration_curve(
    y_true,
    y_prob,
    threshold: float,
    png_path: Path,
    bins: int = 10,
) -> dict:
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=bins, strategy="uniform")

    plt.figure(figsize=(5, 4))
    plt.plot(mean_pred, frac_pos, marker="o", label="Validation reliability")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed frequency")
    plt.title("Validation Calibration")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.close()

    info = {
        "mean_pred": mean_pred.tolist(),
        "frac_pos": frac_pos.tolist(),
        "threshold": threshold,
    }
    return info


def plot_pr_with_threshold(
    y_true,
    y_prob,
    threshold: float,
    png_path: Path,
    split_name: str,
) -> dict:
    precision, recall, thresh = precision_recall_curve(y_true, y_prob)
    ap_value = average_precision_score(y_true, y_prob)
    plt.figure(figsize=(5, 4))
    plt.plot(recall, precision, label=f"{split_name} PR (AP={ap_value:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{split_name} Precision-Recall")
    plt.grid(True, alpha=0.3)

    # Find closest threshold in curve
    thresh_extended = np.append(thresh, thresh[-1])
    idx = np.argmin(np.abs(thresh_extended - threshold))
    chosen_precision = precision[idx]
    chosen_recall = recall[idx]
    plt.scatter([chosen_recall], [chosen_precision], color="red", label="Operating point")
    plt.annotate(
        f"P={chosen_precision:.2f}, R={chosen_recall:.2f}",
        (chosen_recall, chosen_precision),
        textcoords="offset points",
        xytext=(10, -10),
    )

    plt.legend()
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.close()

    return {
        "precision_curve": precision.tolist(),
        "recall_curve": recall.tolist(),
        "chosen_precision": float(chosen_precision),
        "chosen_recall": float(chosen_recall),
        "average_precision": float(ap_value),
    }


def plot_feature_importance(
    feature_names,
    importances,
    png_path: Path,
    top_n: int = 25,
) -> None:
    pairs = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
    names, scores = zip(*pairs[:top_n]) if pairs else ([], [])

    plt.figure(figsize=(8, max(3, len(names) * 0.35)))
    plt.barh(range(len(names)), scores)
    plt.yticks(range(len(names)), names)
    plt.gca().invert_yaxis()
    plt.xlabel("Feature importance")
    plt.title("XGBoost feature importance (gain)")
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.close()

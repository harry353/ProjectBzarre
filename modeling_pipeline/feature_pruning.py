from __future__ import annotations

import numpy as np
import pandas as pd


def select_pruned_features(
    X: pd.DataFrame,
    feature_cols: list[str],
    importances: np.ndarray,
    target_features: int = 40,
    corr_threshold: float = 0.85,
) -> list[str]:
    """
    HARD feature pruning.

    Guarantees:
    - Output feature count == target_features (or fewer if impossible)
    - Never increases feature count
    - Collapses correlated features
    - Uses importance as the final authority

    This WILL prune. No guessing.
    """

    importances = np.asarray(importances, dtype=float)

    if importances.size != len(feature_cols):
        return feature_cols

    if len(feature_cols) <= target_features:
        return feature_cols

    # ---------------------------------
    # Stage 1: Correlation collapse
    # ---------------------------------
    corr = X[feature_cols].corr().abs()
    remaining = list(range(len(feature_cols)))
    kept = []

    while remaining:
        i = remaining.pop(0)
        cluster = [i]

        for j in remaining[:]:
            if corr.iloc[i, j] >= corr_threshold:
                cluster.append(j)
                remaining.remove(j)

        # Keep most important in cluster
        best = max(cluster, key=lambda k: importances[k])
        kept.append(best)

    kept = sorted(set(kept))

    # ---------------------------------
    # Stage 2: HARD importance cutoff
    # ---------------------------------
    kept_sorted = sorted(
        kept,
        key=lambda i: importances[i],
        reverse=True,
    )

    kept_final = kept_sorted[:target_features]

    return [feature_cols[i] for i in sorted(kept_final)]

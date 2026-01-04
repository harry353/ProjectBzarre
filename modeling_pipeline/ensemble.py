from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import optuna
from xgboost import XGBClassifier


def train_top_trial_ensemble(
    study: optuna.Study,
    top_k: int,
    base_kwargs: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    sample_weight: np.ndarray,
    model_dir: Path,
) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    completed_trials = [
        t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None
    ]
    if not completed_trials:
        raise RuntimeError("Optuna produced no completed trials to ensemble.")
    sorted_trials = sorted(completed_trials, key=lambda t: t.value, reverse=True)
    top_trials = sorted_trials[:top_k]

    val_predictions: List[np.ndarray] = []
    test_predictions: List[np.ndarray] = []
    trial_summaries: List[Dict] = []

    for idx, trial in enumerate(top_trials):
        params = trial.params.copy()
        params["random_state"] = base_kwargs["random_state"] + idx
        model_kwargs = {**base_kwargs, **params}

        model = XGBClassifier(**model_kwargs)
        model.fit(X_train, y_train, sample_weight=sample_weight)

        val_predictions.append(model.predict_proba(X_val)[:, 1])
        test_predictions.append(model.predict_proba(X_test)[:, 1])

        model_path = model_dir / f"xgb_optuna_trial_{trial.number}.json"
        model.get_booster().save_model(model_path)

        trial_summaries.append(
            {
                "trial_number": trial.number,
                "validation_average_precision": float(trial.value),
                "params": trial.params,
                "model_path": str(model_path),
            }
        )

    val_prob = np.mean(val_predictions, axis=0)
    test_prob = np.mean(test_predictions, axis=0)
    return val_prob, test_prob, trial_summaries

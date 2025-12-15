from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_absolute_error, mean_squared_error


DATA_DIR = Path(__file__).resolve().parent / "data"
OUTPUT_DIR = Path(__file__).resolve().parent / "catboost_outputs"
MODEL_BIN_PATH = OUTPUT_DIR / "final_catboost_model.cbm"
MODEL_JSON_PATH = OUTPUT_DIR / "final_catboost_model.json"
HYPERPARAMS_PATH = OUTPUT_DIR / "final_hyperparameters.json"
FEATURES_PATH = OUTPUT_DIR / "pruned_features.json"
METRICS_PATH = OUTPUT_DIR / "final_metrics.json"
PREDICTIONS_PATH = OUTPUT_DIR / "test_predictions.parquet"

BASE_PARAMS = {
    "loss_function": "MAE",
    "depth": 4,
    "learning_rate": 0.05,
    "iterations": 900,
    "l2_leaf_reg": 3,
    "task_type": "CPU",
    "allow_writing_files": False,
    "verbose": False,
}
SEED = 1337


def load_splits() -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    train_df = pd.read_parquet(DATA_DIR / "train_h6.parquet")
    val_df = pd.read_parquet(DATA_DIR / "validation_h6.parquet")
    test_df = pd.read_parquet(DATA_DIR / "test_h6.parquet")

    target_col = [c for c in train_df.columns if c.startswith("target_dst_h")][0]
    feature_cols = [c for c in train_df.columns if c != target_col]

    for name, df in [("Train", train_df), ("Validation", val_df), ("Test", test_df)]:
        if df[feature_cols].isna().any().any():
            raise ValueError(f"{name} features contain NaNs.")
        if df[target_col].isna().any():
            raise ValueError(f"{name} target contains NaNs.")

    X_train, y_train = train_df[feature_cols].copy(), train_df[target_col].copy()
    X_val, y_val = val_df[feature_cols].copy(), val_df[target_col].copy()
    X_test, y_test = test_df[feature_cols].copy(), test_df[target_col].copy()

    print(f"Loaded datasets: Train {X_train.shape}, Val {X_val.shape}, Test {X_test.shape}")
    return X_train, y_train, X_val, y_val, X_test, y_test


def train_and_eval(
    params: Dict[str, object],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    feature_names: Sequence[str],
    eval_name: str,
    allow_early_stopping: bool = False,
) -> Dict[str, object]:
    train_pool = Pool(X_train, y_train, feature_names=list(feature_names))
    val_pool = Pool(X_val, y_val, feature_names=list(feature_names))

    model = CatBoostRegressor(**params)
    fit_params = {}
    if allow_early_stopping:
        fit_params.update({"eval_set": val_pool, "use_best_model": True})
    else:
        fit_params.update({"eval_set": val_pool})

    model.fit(train_pool, **fit_params)
    train_pred = model.predict(train_pool)
    val_pred = model.predict(val_pool)
    train_mae = mean_absolute_error(y_train, train_pred)
    val_mae = mean_absolute_error(y_val, val_pred)

    result = {
        "name": eval_name,
        "params": params.copy(),
        "train_mae": float(train_mae),
        "val_mae": float(val_mae),
        "model": model,
    }
    print(f"{eval_name}: train MAE={train_mae:.4f}, val MAE={val_mae:.4f}, iterations={model.tree_count_}")
    return result


def stage_a_learning_rate(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    feature_names: Sequence[str],
) -> List[Dict[str, object]]:
    learning_rates = [0.03, 0.04, 0.05, 0.06, 0.07]
    results = []
    for lr in learning_rates:
        params = BASE_PARAMS | {"learning_rate": lr, "random_seed": SEED}
        res = train_and_eval(
            params,
            X_train,
            y_train,
            X_val,
            y_val,
            feature_names,
            eval_name=f"StageA_lr_{lr}",
            allow_early_stopping=False,
        )
        results.append(res)
    results.sort(key=lambda x: x["val_mae"])
    print("Stage A top candidates:")
    for item in results[:3]:
        print(f"  lr={item['params']['learning_rate']:.3f}, val_mae={item['val_mae']:.4f}")
    return results


def stage_b_iterations(
    base_lrs: Iterable[float],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    feature_names: Sequence[str],
) -> List[Dict[str, object]]:
    results = []
    for lr in base_lrs:
        for iters in range(600, 2100, 100):
            params = BASE_PARAMS | {
                "learning_rate": lr,
                "iterations": iters,
                "od_wait": min(150, max(50, iters // 10)),
                "od_type": "Iter",
                "random_seed": SEED,
            }
            res = train_and_eval(
                params,
                X_train,
                y_train,
                X_val,
                y_val,
                feature_names,
                eval_name=f"StageB_lr{lr}_it{iters}",
                allow_early_stopping=True,
            )
            res["iterations_used"] = res["model"].tree_count_
            results.append(res)
    results.sort(key=lambda x: x["val_mae"])
    print("Stage B best results:")
    for item in results[:3]:
        print(
            f"  lr={item['params']['learning_rate']} iters={item['params']['iterations']} "
            f"used={item['iterations_used']} val_mae={item['val_mae']:.4f}"
        )
    return results


def stage_c_depth_search(
    base_params: Dict[str, object],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    feature_names: Sequence[str],
) -> List[Dict[str, object]]:
    depths = [3, 4, 5, 6]
    results = []
    for depth in depths:
        params = base_params | {
            "depth": depth,
            "grow_policy": "SymmetricTree",
            "l2_leaf_reg": base_params.get("l2_leaf_reg", 3),
            "border_count": 254 if depth <= 4 else 128,
            "random_seed": SEED,
        }
        res = train_and_eval(
            params,
            X_train,
            y_train,
            X_val,
            y_val,
            feature_names,
            eval_name=f"StageC_depth_{depth}",
            allow_early_stopping=True,
        )
        res["overfit"] = res["train_mae"] + 0.02 < res["val_mae"]
        results.append(res)
    results = [r for r in results if not r.get("overfit")]
    results.sort(key=lambda x: x["val_mae"])
    print("Stage C filtered results:")
    for item in results[:3]:
        print(f"  depth={item['params']['depth']} val_mae={item['val_mae']:.4f}")
    return results


def stage_d_regularization(
    template_params: Dict[str, object],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    feature_names: Sequence[str],
) -> List[Dict[str, object]]:
    l2_list = [1, 3, 5, 7, 10]
    bagging_temps = [0.0, 0.25, 0.5, 1.0]
    random_strengths = [0.5, 1.0, 2.0, 5.0]
    results = []
    for l2 in l2_list:
        for bag_temp in bagging_temps:
            for rs in random_strengths:
                params = template_params | {
                    "l2_leaf_reg": l2,
                    "bagging_temperature": bag_temp,
                    "random_strength": rs,
                    "random_seed": SEED,
                }
                res = train_and_eval(
                    params,
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                    feature_names,
                    eval_name=f"StageD_l2{l2}_bag{bag_temp}_rs{rs}",
                    allow_early_stopping=True,
                )
                results.append(res)
    results.sort(key=lambda x: x["val_mae"])
    print("Stage D top results:")
    for item in results[:5]:
        params = item["params"]
        print(
            f"  l2={params['l2_leaf_reg']}, bag_temp={params.get('bagging_temperature', 0)}, "
            f"random_strength={params.get('random_strength', 1)}, val_mae={item['val_mae']:.4f}"
        )
    return results[:5]


def stage_e_feature_pruning(
    base_params: Dict[str, object],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    protected_prefixes: Sequence[str],
) -> Tuple[pd.Index, Dict[str, object], List[Dict[str, float]]]:
    current_features = X_train.columns.copy()
    history: List[Dict[str, float]] = []
    best_val = float("inf")
    best_features = current_features
    params = base_params.copy()
    improved = True

    while improved and len(current_features) > 50:
        result = train_and_eval(
            params,
            X_train[current_features],
            y_train,
            X_val[current_features],
            y_val,
            current_features,
            eval_name=f"StageE_prune_{len(current_features)}",
            allow_early_stopping=True,
        )
        model = result["model"]
        val_mae = result["val_mae"]
        importance = model.get_feature_importance(
            Pool(X_val[current_features], y_val, feature_names=list(current_features)),
            type="PredictionValuesChange",
        )
        importance_dict = dict(zip(current_features, importance))
        history.append(dict(sorted(importance_dict.items(), key=lambda kv: kv[1], reverse=True)[:10]))
        if val_mae + 0.001 < best_val:
            best_val = val_mae
            best_features = current_features
            print(f"  Improvement detected with {len(current_features)} features, val_mae={val_mae:.4f}")
        else:
            print("  No improvement from pruning; stopping.")
            break

        num_remove = max(1, int(0.08 * len(current_features)))
        sorted_feats = sorted(importance_dict.items(), key=lambda kv: kv[1])
        to_drop = []
        for feat, _ in sorted_feats:
            if any(feat.startswith(prefix) for prefix in protected_prefixes):
                continue
            to_drop.append(feat)
            if len(to_drop) >= num_remove:
                break
        if not to_drop:
            print("  No removable features found; stopping pruning.")
            break
        current_features = current_features.difference(to_drop)
        print(f"  Pruned {len(to_drop)} features; remaining {len(current_features)}.")

    return best_features, params, history


def stage_f_ensembling(
    params: Dict[str, object],
    feature_names: Sequence[str],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> Tuple[List[CatBoostRegressor], float]:
    seeds = [SEED + i * 17 for i in range(5)]
    models = []
    val_preds = []

    for idx, seed in enumerate(seeds, 1):
        local_params = params | {"random_seed": seed}
        res = train_and_eval(
            local_params,
            X_train,
            y_train,
            X_val,
            y_val,
            feature_names,
            eval_name=f"StageF_seed_{seed}",
            allow_early_stopping=True,
        )
        models.append(res["model"])
        val_preds.append(res["model"].predict(X_val))
        print(f"  Model {idx} val_mae={res['val_mae']:.4f}")

    ensemble_pred = np.mean(val_preds, axis=0)
    ensemble_mae = mean_absolute_error(y_val, ensemble_pred)
    print(f"Ensemble validation MAE={ensemble_mae:.4f}")
    return models, ensemble_mae


def evaluate_final_model(
    model: CatBoostRegressor,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, object]:
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds, squared=False)
    residuals = y_test - preds
    worst_idx = np.argmax(np.abs(residuals))
    metrics = {
        "test_mae": float(mae),
        "test_rmse": float(rmse),
        "worst_error": float(residuals.iloc[worst_idx]),
        "worst_timestamp": int(worst_idx),
        "residual_mean": float(residuals.mean()),
        "residual_std": float(residuals.std()),
    }
    print(f"Final Test MAE={mae:.4f}, RMSE={rmse:.4f}")
    return metrics, preds, residuals


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    X_train, y_train, X_val, y_val, X_test, y_test = load_splits()
    feature_names = X_train.columns

    stage_a_results = stage_a_learning_rate(X_train, y_train, X_val, y_val, feature_names)
    top_lrs = [res["params"]["learning_rate"] for res in stage_a_results[:2]]

    stage_b_results = stage_b_iterations(top_lrs, X_train, y_train, X_val, y_val, feature_names)
    best_stage_b = stage_b_results[0]

    stage_c_results = stage_c_depth_search(best_stage_b["params"], X_train, y_train, X_val, y_val, feature_names)
    best_stage_c = stage_c_results[0]

    stage_d_results = stage_d_regularization(best_stage_c["params"], X_train, y_train, X_val, y_val, feature_names)
    best_stage_d = stage_d_results[0]

    protected_prefixes = ["Dst", "dst", "Dst_lag", "dst_lag"]
    pruned_features, pruned_params, importance_history = stage_e_feature_pruning(
        best_stage_d["params"],
        X_train,
        y_train,
        X_val,
        y_val,
        protected_prefixes,
    )

    ensemble_models, ensemble_mae = stage_f_ensembling(
        pruned_params,
        pruned_features,
        X_train[pruned_features],
        y_train,
        X_val[pruned_features],
        y_val,
    )

    best_model = min(ensemble_models, key=lambda m: mean_absolute_error(y_val, m.predict(X_val[pruned_features])))
    if ensemble_mae + 0.002 < mean_absolute_error(y_val, best_model.predict(X_val[pruned_features])):
        print("Using ensemble average for final predictions.")
        final_model = None
    else:
        final_model = best_model

    combined_X = pd.concat([X_train[pruned_features], X_val[pruned_features]], axis=0)
    combined_y = pd.concat([y_train, y_val], axis=0)
    final_params = pruned_params | {"random_seed": SEED + 999}
    final_model = CatBoostRegressor(**final_params)
    final_model.fit(Pool(combined_X, combined_y, feature_names=list(pruned_features)), verbose=False)

    metrics, preds, residuals = evaluate_final_model(final_model, X_test[pruned_features], y_test)

    final_model.save_model(MODEL_BIN_PATH, format="cbm")
    final_model.save_model(MODEL_JSON_PATH, format="json")
    with open(HYPERPARAMS_PATH, "w", encoding="utf-8") as fp:
        json.dump(final_params, fp, indent=2)
    with open(FEATURES_PATH, "w", encoding="utf-8") as fp:
        json.dump(list(pruned_features), fp, indent=2)

    metrics["residual_mean"] = float(residuals.mean())
    metrics["residual_std"] = float(residuals.std())
    metrics["max_residual"] = float(residuals.max())
    metrics["min_residual"] = float(residuals.min())
    with open(METRICS_PATH, "w", encoding="utf-8") as fp:
        json.dump(metrics, fp, indent=2)

    pd.DataFrame(
        {
            "pred": preds,
            "actual": y_test,
            "residual": residuals,
        }
    ).to_parquet(PREDICTIONS_PATH, index=False)
    print(f"Saved outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

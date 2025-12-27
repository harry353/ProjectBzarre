from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
import time


PIPELINE_ROOT = Path(__file__).resolve().parent
PIPELINE_HORIZON_HOURS = 4
PIPELINE_N_JOBS = 12
FEATURE_EXCLUDE_PATTERNS = []
TARGET_COLUMN_TOKEN = "main_phase"  # options: "severity", "main_phase", "ssc", or direct column name

_TARGET_ALIAS = {
    "severity": "severity_label",
    "storm": "severity_label",
    "main": "main_phase_label",
    "main_phase": "main_phase_label",
    "ssc": "ssc_label",
}


def _resolve_target_column(token: str) -> str:
    key = (token or "").lower()
    return _TARGET_ALIAS.get(key, token)

OPTUNA_TRIALS = {
    "stageA": 1,
    "stageB": 2,
    "stageC": 2,
    "stageD": 3,
    "stageE": 1,
}

# OPTUNA_TRIALS = {
#     "stageA": 50,
#     "stageB": 60,
#     "stageC": 60,
#     "stageD": 80,
#     "stageE": 20,
# }

STEPS = [
    (PIPELINE_ROOT / "optuna_studies" / "stageA_learning_rate" / "run_stageA_optuna.py", "stageA"),
    (PIPELINE_ROOT / "optuna_studies" / "stageB_iterations" / "run_stageB_optuna.py", "stageB"),
    (PIPELINE_ROOT / "optuna_studies" / "stageC_tree_params" / "run_stageC_optuna.py", "stageC"),
    (PIPELINE_ROOT / "optuna_studies" / "stageD_regularization" / "run_stageD_optuna.py", "stageD"),
    (PIPELINE_ROOT / "optuna_studies" / "stageE_scale_pos_weight" / "run_stageE_optuna.py", "stageE"),
    (PIPELINE_ROOT / "feature_pruning" / "run_feature_pruning.py", None),
    (PIPELINE_ROOT / "models" / "ensembles" / "run_ensembling.py", None),
    (PIPELINE_ROOT / "models" / "final_model" / "run_final_training.py", None),
]


def run_step(script: Path, stage_key: str | None, horizon: int, n_jobs: int, target_column: str) -> None:
    print(f"[RUN] {script}")
    env = os.environ.copy()
    env["PIPELINE_HORIZON"] = str(horizon)
    env["PIPELINE_N_JOBS"] = str(n_jobs)
    env["PIPELINE_FEATURE_EXCLUDES"] = json.dumps(FEATURE_EXCLUDE_PATTERNS)
    env["PIPELINE_TARGET_COLUMN"] = target_column
    cmd = [sys.executable, str(script)]
    if stage_key is not None:
        trials = OPTUNA_TRIALS.get(stage_key)
        if trials is not None:
            cmd.extend(["--trials", str(trials)])
    result = subprocess.run(cmd, check=False, env=env)
    if result.returncode != 0:
        raise RuntimeError(f"Step failed: {script}")
    print(f"[OK] {script}")


def main() -> None:
    start_time = time.time()
    target_column = _resolve_target_column(TARGET_COLUMN_TOKEN)
    print(f"[INFO] Using target column: {target_column}")
    for script, stage_key in STEPS:
        run_step(script, stage_key, PIPELINE_HORIZON_HOURS, PIPELINE_N_JOBS, target_column)
    elapsed = time.time() - start_time
    print(f"[ALL DONE] Full modeling pipeline completed in {elapsed / 60:.2f} minutes.")


if __name__ == "__main__":
    main()

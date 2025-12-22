from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PIPELINE_ROOT = Path(__file__).resolve().parent
BASE_DIR = PIPELINE_ROOT.parent
OPTUNA_TRIALS = {
    "stageA": 1,
    "stageB": 2,
    "stageC": 2,
    "stageD": 3,
    "stageE": 20,
}

# OPTUNA_TRIALS = {
#     "stageA": 50,
#     "stageB": 60,
#     "stageC": 60,
#     "stageD": 80,
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


def run_step(script: Path, stage_key: str | None) -> None:
    print(f"[RUN] {script}")
    cmd = [sys.executable, str(script)]
    if stage_key is not None:
        trials = OPTUNA_TRIALS.get(stage_key)
        if trials is not None:
            cmd.extend(["--trials", str(trials)])
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Step failed: {script}")
    print(f"[OK] {script}")


def main() -> None:
    for script, stage_key in STEPS:
        run_step(script, stage_key)
    print("[ALL DONE] Full modeling pipeline completed.")


if __name__ == "__main__":
    main()

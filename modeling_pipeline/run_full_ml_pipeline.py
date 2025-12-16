from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PIPELINE_ROOT = Path(__file__).resolve().parent
BASE_DIR = PIPELINE_ROOT.parent
STEPS = [
    PIPELINE_ROOT / "optuna_studies" / "stageA_learning_rate" / "run_stageA_optuna.py",
    PIPELINE_ROOT / "optuna_studies" / "stageB_iterations" / "run_stageB_optuna.py",
    PIPELINE_ROOT / "optuna_studies" / "stageC_tree_params" / "run_stageC_optuna.py",
    PIPELINE_ROOT / "optuna_studies" / "stageD_regularization" / "run_stageD_optuna.py",
    PIPELINE_ROOT / "feature_pruning" / "run_feature_pruning.py",
    PIPELINE_ROOT / "models" / "ensembles" / "run_ensembling.py",
    PIPELINE_ROOT / "models" / "final_model" / "run_final_training.py",
]


def run_step(script: Path) -> None:
    print(f"[RUN] {script}")
    result = subprocess.run([sys.executable, str(script)], check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Step failed: {script}")
    print(f"[OK] {script}")


def main() -> None:
    for step in STEPS:
        run_step(step)
    print("[ALL DONE] Full modeling pipeline completed.")


if __name__ == "__main__":
    main()

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


# Resolve the project root two levels above this file, independent of the working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Ordered pipeline stages — each script depends on the output of the previous one.
SCRIPTS = [
    PROJECT_ROOT / "regression_pipeline" / "diagnose_log_candidates.py",        # Stage 1: analyse features, emit log_candidates.csv
    PROJECT_ROOT / "regression_pipeline" / "apply_log_zscore_transforms.py",    # Stage 2: apply log1p + z-score, emit transformed_features.db
    PROJECT_ROOT / "regression_pipeline" / "apply_pca_transforms.py",           # Stage 3: fit & apply PCA, emit pca_features.db
    PROJECT_ROOT / "regression_pipeline" / "rf_student_t_model" / "train_student_t_rf.py",  # Stage 4: train the Student-t RF model
]


def run_script(path: Path) -> None:
    # Fail early with a clear message if the script file is missing.
    if not path.exists():
        raise FileNotFoundError(f"Script not found: {path}")
    print(f"[RUN] {path}")
    # check=True halts the pipeline immediately if a stage exits with a non-zero code.
    subprocess.run([sys.executable, str(path)], check=True)


def main() -> None:
    # Execute stages in declaration order — order is significant.
    for script in SCRIPTS:
        run_script(script)
    print("[DONE] Regression pipeline completed.")


if __name__ == "__main__":
    main()

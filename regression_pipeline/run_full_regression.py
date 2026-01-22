from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = [
    PROJECT_ROOT / "regression_pipeline" / "diagnose_log_candidates.py",
    PROJECT_ROOT / "regression_pipeline" / "apply_log_zscore_transforms.py",
    PROJECT_ROOT / "regression_pipeline" / "apply_pca_transforms.py",
    PROJECT_ROOT / "regression_pipeline" / "rf_student_t_model" / "train_student_t_rf.py",
]


def run_script(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Script not found: {path}")
    print(f"[RUN] {path}")
    subprocess.run([sys.executable, str(path)], check=True)


def main() -> None:
    for script in SCRIPTS:
        run_script(script)
    print("[DONE] Regression pipeline completed.")


if __name__ == "__main__":
    main()

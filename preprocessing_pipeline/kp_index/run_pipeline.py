from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

# Resolve absolute path of the current script and search for project root (space_weather_api.py)
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE
for parent in THIS_FILE.parents:
    if (parent / "space_weather_api.py").exists():
        PROJECT_ROOT = parent
        break
else:
    PROJECT_ROOT = PROJECT_ROOT.parent

# Inject project root into system path to allow local module imports
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SOURCE_DIR = THIS_FILE.parent


# Executes a given Python script in a separate sub-process with optional arguments
def _run_stage(script: Path, extra_args: list[str] | None = None) -> None:
    cmd = [sys.executable, str(script)]
    if extra_args:
        cmd.extend(extra_args)
    subprocess.run(cmd, check=True)


# Orchestrates the 6-stage Kp-Index preprocessing pipeline
def main() -> None:
    # Execution sequence: Raw -> Hourly -> Diagnostics -> Filters -> Imputation -> Splits -> Features
    stages = [
        SOURCE_DIR / "1_averaging" / "build_hourly.py",
        SOURCE_DIR / "2_missingness" / "plot_missingness.py",
        SOURCE_DIR / "3_hard_filtering" / "apply_filters.py",
        SOURCE_DIR / "4_imputation" / "run_imputation.py",
        SOURCE_DIR / "5_train_test_split" / "create_splits.py",
        SOURCE_DIR / "6_engineered_features" / "engineer_features.py",
    ]
    
    # Process each script sequentially; halts on first failure
    for script in stages:
        _run_stage(script)

    # Final consolidated feature database for the Kp module
    final_db = SOURCE_DIR / "kp_fin.db"
    print(f"[OK] KP final database available at {final_db}")


if __name__ == "__main__":
    main()

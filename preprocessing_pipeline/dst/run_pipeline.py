from __future__ import annotations

import sys
from pathlib import Path

# Resolve absolute path of the current script to locate the project root
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE
# Traverse upwards to find the root directory (marked by space_weather_api.py)
for parent in THIS_FILE.parents:
    if (parent / "space_weather_api.py").exists():
        PROJECT_ROOT = parent
        break
else:
    PROJECT_ROOT = PROJECT_ROOT.parent

# Ensure the project root is in the system path for local imports
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import subprocess

SOURCE_DIR = THIS_FILE.parent


# Helper to run a specific pipeline stage in an isolated subprocess
def _run_stage(script: Path, extra_args: list[str] | None = None) -> None:
    cmd = [sys.executable, str(script)]
    if extra_args:
        cmd.extend(extra_args)
    subprocess.run(cmd, check=True)


def main() -> None:
    # Definition of the sequential steps required for DST preprocessing
    stages = [
        SOURCE_DIR / "1_averaging" / "build_hourly.py",           # Aggregate to hourly frequency
        SOURCE_DIR / "2_missingness" / "plot_missingness.py",     # Generate diagnostic visuals
        SOURCE_DIR / "3_hard_filtering" / "apply_filters.py",     # Remove invalid samples
        SOURCE_DIR / "4_imputation" / "run_imputation.py",        # Fill data gaps
        SOURCE_DIR / "5_train_test_split" / "create_splits.py",    # Partition data partitions
        SOURCE_DIR / "6_engineered_features" / "engineer_features.py", # Generate models inputs
    ]
    
    # Iterate and execute each stage; failure in any stage halts the pipeline
    for script in stages:
        _run_stage(script)

    final_db = SOURCE_DIR / "dst_fin.db"
    print(f"[OK] DST final database available at {final_db}")


if __name__ == "__main__":
    main()

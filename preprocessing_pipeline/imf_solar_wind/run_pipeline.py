from __future__ import annotations

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


# Executes a specific pipeline stage script in a separate Python process
def _run_stage(script: Path) -> None:
    subprocess.run([sys.executable, str(script)], check=True)


# Orchestrates the 7-stage IMF and Solar Wind preprocessing pipeline
def main() -> None:
    # Defined sequence of operations: Raw -> Averaged -> Combined -> Filtered -> Imputed -> Split -> Features
    stages = [
        SOURCE_DIR / "1_averaging" / "build_hourly.py",
        SOURCE_DIR / "2_concatenating_combining" / "combine_instruments.py",
        SOURCE_DIR / "3_missingness" / "plot_missingness.py",
        SOURCE_DIR / "4_hard_filtering" / "apply_filters.py",
        SOURCE_DIR / "5_imputation" / "run_imputation.py",
        SOURCE_DIR / "6_train_test_split" / "create_splits.py",
        SOURCE_DIR / "7_engineered_features" / "engineer_features.py",
    ]

    # Process each stage sequentially; failure in any stage halts the pipeline
    for script in stages:
        _run_stage(script)

    # Final consolidated feature matrix location
    final_db = SOURCE_DIR / "imf_solar_wind_fin.db"
    print(f"[OK] IMF + solar wind preprocessing pipeline completed. Final DB located at {final_db}")


if __name__ == "__main__":
    main()

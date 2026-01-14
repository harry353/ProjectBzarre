from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE
for parent in THIS_FILE.parents:
    if (parent / "space_weather_api.py").exists():
        PROJECT_ROOT = parent
        break
else:
    PROJECT_ROOT = PROJECT_ROOT.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SOURCE_DIR = THIS_FILE.parent


def _run_stage(script: Path) -> None:
    subprocess.run([sys.executable, str(script)], check=True)


def main() -> None:
    skip_splits = os.environ.get("XRS_SKIP_SPLITS") == "1"

    stages = [
        SOURCE_DIR / "1_averaging" / "build_hourly.py",
        SOURCE_DIR / "2_missingness" / "plot_missingness.py",
        SOURCE_DIR / "3_hard_filtering" / "apply_filters.py",
        SOURCE_DIR / "4_imputation" / "run_imputation.py",
        SOURCE_DIR / "5_feature_engineering" / "engineer_features.py",
        SOURCE_DIR / "6_aggregate" / "create_aggregate_features.py",
    ]

    if skip_splits:
        os.environ["XRS_INFERENCE_MODE"] = "1"
        stages.append(SOURCE_DIR / "8_normalization" / "normalize.py")
    else:
        os.environ.pop("XRS_INFERENCE_MODE", None)
        stages.extend(
            [
                SOURCE_DIR / "7_train_test_split" / "create_splits.py",
                SOURCE_DIR / "8_normalization" / "normalize.py",
            ]
        )

    for script in stages:
        _run_stage(script)

    print("[OK] X-ray flux preprocessing pipeline completed. Final DB located at xray_flux_fin.db")


if __name__ == "__main__":
    main()

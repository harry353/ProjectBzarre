from __future__ import annotations

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
    stages = [
        SOURCE_DIR / "1_combining" / "combine_goes_flares.py",
        SOURCE_DIR / "2_hard_filtering" / "apply_filters.py",
        SOURCE_DIR / "3_engineered_features" / "engineer_features.py",
        SOURCE_DIR / "4_train_test_split" / "create_splits.py",
        SOURCE_DIR / "5_normalization" / "normalize.py",
    ]

    for script in stages:
        _run_stage(script)

    print("[OK] Flare preprocessing pipeline completed. Final DB located at flare_fin.db")


if __name__ == "__main__":
    main()

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
    cmd = [sys.executable, str(script)]
    subprocess.run(cmd, check=True)


def main() -> None:
    stages = [
        SOURCE_DIR / "1_hard_filtering" / "apply_filters.py",
        SOURCE_DIR / "2_engineered_features" / "engineer_features.py",
        SOURCE_DIR / "3_aggregate" / "create_aggregate_features.py",
        SOURCE_DIR / "4_train_test_split" / "create_splits.py",
        SOURCE_DIR / "5_normalization" / "normalize.py",
    ]

    for script in stages:
        _run_stage(script)

    final_db = SOURCE_DIR / "5_normalization" / "radio_flux_agg_eng_split_norm.db"
    destination = SOURCE_DIR / "radio_flux_fin.db"
    destination.write_bytes(final_db.read_bytes())
    print(f"[OK] Radio flux pipeline complete. Final DB: {final_db} (copied to {destination})")


if __name__ == "__main__":
    main()

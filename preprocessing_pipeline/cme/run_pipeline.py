from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# Resolve the absolute path of the current script
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE
# Ascend the directory tree to locate the project root (identified by space_weather_api.py)
for parent in THIS_FILE.parents:
    if (parent / "space_weather_api.py").exists():
        PROJECT_ROOT = parent
        break
else:
    PROJECT_ROOT = PROJECT_ROOT.parent

# Add PROJECT_ROOT to the system path for module imports
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SOURCE_DIR = THIS_FILE.parent


# Helper function to execute a specific pipeline stage via a subprocess
def _run_stage(script: Path) -> None:
    subprocess.run([sys.executable, str(script)], check=True)


def main() -> None:
    # Sequence of scripts that define the CME preprocessing workflow
    stages = [
        SOURCE_DIR / "1_train_test_split" / "create_splits.py",
        SOURCE_DIR / "2_engineered_features" / "engineer_features.py",
    ]

    # Execute each stage sequentially; if any fails, the process terminates
    for script in stages:
        _run_stage(script)

    final_db = SOURCE_DIR / "cme_fin.db"
    print(f"[OK] CME preprocessing pipeline completed. Final DB located at {final_db}")


if __name__ == "__main__":
    main()

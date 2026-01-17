from __future__ import annotations

import subprocess
import sys
from pathlib import Path
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SCRIPTS = [
    PROJECT_ROOT / "inference" / "update_space_weather_last_1944h.py",
    PROJECT_ROOT / "inference" / "run_preprocessing_on_latest_1944h.py",
    PROJECT_ROOT / "inference" / "create_horizon_vector.py",
    PROJECT_ROOT / "inference" / "run_horizon_models.py",
]


def _run(script: Path) -> None:
    if not script.exists():
        raise FileNotFoundError(f"Script not found: {script}")
    print(f"[RUN] {script}")
    subprocess.run([sys.executable, str(script)], check=True)


def main() -> None:
    start = time.time()
    for script in SCRIPTS:
        _run(script)
    # Clean up intermediate DBs
    for fname in ("horizon_vector.db", "inference_vector.db"):
        path = PROJECT_ROOT / "inference" / fname
        if path.exists():
            try:
                path.unlink()
                print(f"[CLEANUP] Removed {path}")
            except Exception as exc:
                print(f"[WARN] Could not remove {path}: {exc}")
    elapsed = time.time() - start
    print(f"[OK] Inference pipeline completed in {elapsed:.2f} seconds.")


if __name__ == "__main__":
    main()

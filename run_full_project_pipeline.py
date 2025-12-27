from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

PREPROCESS_SCRIPT = PROJECT_ROOT / "preprocessing_pipeline" / "run_full_preprocessing_pipeline.py"
MODEL_SCRIPT = PROJECT_ROOT / "modeling_pipeline_bin" / "run_full_ml_pipeline.py"


def _run(script: Path) -> None:
    if not script.exists():
        raise FileNotFoundError(f"Required script missing: {script}")
    print(f"[RUN] {script}")
    subprocess.run([sys.executable, str(script)], check=True)
    print(f"[OK] {script}")


def main() -> None:
    _run(PREPROCESS_SCRIPT)
    _run(MODEL_SCRIPT)
    print("[ALL DONE] Full preprocessing + modeling pipeline completed.")


if __name__ == "__main__":
    main()

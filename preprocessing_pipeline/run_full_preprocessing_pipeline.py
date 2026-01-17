from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve()
for parent in PROJECT_ROOT.parents:
    if (parent / "space_weather_api.py").exists():
        PROJECT_ROOT = parent
        break
else:
    PROJECT_ROOT = PROJECT_ROOT.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PIPELINE_DIR = PROJECT_ROOT / "preprocessing_pipeline"

RUNNERS = [
    PIPELINE_DIR / "dst" / "run_pipeline.py",
    PIPELINE_DIR / "kp_index" / "run_pipeline.py",
    PIPELINE_DIR / "sunspot_number" / "run_pipeline.py",
    PIPELINE_DIR / "cme" / "run_pipeline.py",
    PIPELINE_DIR / "imf_solar_wind" / "run_pipeline.py",
    # PIPELINE_DIR / "xray_flux" / "run_pipeline.py",
    PIPELINE_DIR / "radio_flux" / "run_pipeline.py",
    PIPELINE_DIR / "labels" / "hazard_label" / "build_supervised_targets.py",
]

FINAL_SCRIPTS = [
    PIPELINE_DIR / "check_multicolinearity" / "merge_features.py",
]

DEFAULT_WINDOWS = {
    "train_start": "1999-01-01",
    "train_end": "2016-12-31",
    "validation_start": "2017-01-01",
    "validation_end": "2020-12-31",
    "test_start": "2021-01-01",
    "test_end": "2025-12-31",
}

PIPELINE_ANCHOR_HOURS = 0
PIPELINE_AGG_FREQUENCY = "8h"


def _run(script: Path) -> None:
    if not script.exists():
        print(f"[WARN] Skipping missing pipeline runner: {script}")
        return
    print(f"[INFO] Running pipeline: {script}")
    subprocess.run([sys.executable, str(script)], check=True)


def _set_split_env(args: argparse.Namespace) -> None:
    os.environ["PREPROC_SPLIT_TRAIN_START"] = args.train_start
    os.environ["PREPROC_SPLIT_TRAIN_END"] = args.train_end
    os.environ["PREPROC_SPLIT_VALIDATION_START"] = args.validation_start
    os.environ["PREPROC_SPLIT_VALIDATION_END"] = args.validation_end
    os.environ["PREPROC_SPLIT_TEST_START"] = args.test_start
    os.environ["PREPROC_SPLIT_TEST_END"] = args.test_end


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all preprocessing pipelines with shared split windows.")
    parser.add_argument("--train-start", default=DEFAULT_WINDOWS["train_start"])
    parser.add_argument("--train-end", default=DEFAULT_WINDOWS["train_end"])
    parser.add_argument("--validation-start", default=DEFAULT_WINDOWS["validation_start"])
    parser.add_argument("--validation-end", default=DEFAULT_WINDOWS["validation_end"])
    parser.add_argument("--test-start", default=DEFAULT_WINDOWS["test_start"])
    parser.add_argument("--test-end", default=DEFAULT_WINDOWS["test_end"])
    args = parser.parse_args()

    _set_split_env(args)
    os.environ["PREPROC_ANCHOR_HOURS"] = str(PIPELINE_ANCHOR_HOURS)
    os.environ["PREPROC_AGG_FREQ"] = PIPELINE_AGG_FREQUENCY

    start = time.perf_counter()
    for runner in RUNNERS:
        _run(runner)
    for script in FINAL_SCRIPTS:
        _run(script)
    elapsed = time.perf_counter() - start
    print(f"[OK] All preprocessing pipelines completed in {elapsed:.1f} seconds.")


if __name__ == "__main__":
    main()

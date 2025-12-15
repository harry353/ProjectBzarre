from __future__ import annotations

import argparse
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

from preprocessing.core.cli import run_source_pipeline

SOURCE_NAME = THIS_FILE.parent.name


def main() -> None:
    parser = argparse.ArgumentParser(
        description=f"Run the full preprocessing pipeline for {SOURCE_NAME}."
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="Forecast horizon for the supervised targets (defaults to the source config).",
    )
    args = parser.parse_args()
    final_path = run_source_pipeline(SOURCE_NAME, horizon=args.horizon)
    print(f"[OK] {SOURCE_NAME} final database saved to {final_path}")


if __name__ == "__main__":
    main()

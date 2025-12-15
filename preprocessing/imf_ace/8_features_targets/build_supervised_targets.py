from __future__ import annotations
import sys
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

import argparse

from preprocessing.core.runner import StageRunner


def main() -> None:
    parser = argparse.ArgumentParser(description="Build supervised targets for imf_ace.")
    parser.add_argument("--horizon", type=int, default=None, help="Forecast horizon in hours.")
    args = parser.parse_args()
    runner = StageRunner("imf_ace")
    runner.build_supervised_targets(horizon=args.horizon)


if __name__ == "__main__":
    main()

from __future__ import annotations

from pathlib import Path

from .config import SourcePipelineConfig, STAGE_NAMES


PREPROCESSING_ROOT = Path(__file__).resolve().parents[1]


class SourcePaths:
    def __init__(self, config: SourcePipelineConfig) -> None:
        self.config = config
        self.base_dir = PREPROCESSING_ROOT / config.name

    def stage_dir(self, stage_number: int) -> Path:
        if stage_number not in STAGE_NAMES:
            raise ValueError(f"Unknown stage number: {stage_number}")
        stage_name = STAGE_NAMES[stage_number]
        path = self.base_dir / f"{stage_number}_{stage_name}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def stage_file(self, stage_number: int, filename: str) -> Path:
        return self.stage_dir(stage_number) / filename

    def hourly_db(self) -> Path:
        return self.stage_file(1, f"{self.config.name}_aver.db")

    def filtered_db(self) -> Path:
        return self.stage_file(3, f"{self.config.name}_filt.db")

    def imputed_db(self) -> Path:
        return self.stage_file(4, f"{self.config.name}_imp.db")

    def features_db(self) -> Path:
        return self.stage_file(5, f"{self.config.name}_eng.db")

    def splits_db(self) -> Path:
        return self.stage_file(6, f"{self.config.name}_split.db")

    def normalized_db(self) -> Path:
        return self.stage_file(7, f"{self.config.name}_norm.db")

    def supervised_db(self, horizon: int) -> Path:
        return self.stage_file(8, f"{self.config.name}_h{horizon}.db")

    def normalization_params(self) -> Path:
        return self.stage_file(7, f"{self.config.name}_normalization.json")

    def missingness_figure(self) -> Path:
        return self.stage_file(2, f"{self.config.name}_missingness.png")

    def parquet_dir(self) -> Path:
        return self.stage_file(8, "parquet_splits")

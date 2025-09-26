# https://www.liander.nl/-/media/files/open-data/historische-15-minuten-bedrijfsmetingen/historische-15-minuten-bedrijfsmetingen.zip


from datetime import timedelta
from enum import StrEnum
from pathlib import Path
from typing import ClassVar, Literal, override

from openstef_beam.benchmarking import BenchmarkTarget
from openstef_beam.benchmarking.target_provider import SimpleTargetProvider
from openstef_core.datasets import VersionedTimeSeriesDataset
import pandas as pd
from pydantic import Field


class Liander2024Group(StrEnum):
    WIND_PARK = "wind_park"
    SOLAR_PARK = "solar_park"
    TRANSFORMER = "transformer"
    STATION_INSTALLATION = "station_installation"
    MV_FEEDER = "mv_feeder"


class Liander2024TargetProvider(SimpleTargetProvider[BenchmarkTarget, list[Liander2024Group]]):
    measurements_path_template: str = Field(
        default="{name}.csv",
        description="Template for target-specific measurement files with {name} placeholder",
    )

    @override
    def get_targets(self, filter_args: list[Liander2024Group] | None = None) -> list[BenchmarkTarget]:
        raise NotImplementedError

    @override
    def get_measurements_for_target(self, target: BenchmarkTarget) -> VersionedTimeSeriesDataset:
        # Load and normalize raw data
        data_raw: pd.DataFrame = pd.read_csv(  # pyright: ignore[reportUnknownMemberType]
            self.get_measurements_path_for_target(target),
            sep=",",
            parse_dates=["datetime"],
            dtype={"load_normalized": float},
        )
        data_raw = data_raw.rename(columns={"datetime": "timestamp", "load_normalized": "load"})
        data_raw["available_at"] = data_raw["timestamp"]

        # Granular measurement data usually comes with a dalay of 2 days.
        if target.group_name in {Liander2024Group.SOLAR_PARK, Liander2024Group.WIND_PARK}:
            data_raw["available_at"] += pd.Timedelta(days=2)

        return VersionedTimeSeriesDataset.from_dataframe(
            data=data_raw,
            sample_interval=timedelta(minutes=15),
            timestamp_column="timestamp",
            available_at_column="available_at",
        )



class Liander2024Benchmark:
    measurements_download_url: ClassVar[str] = "https://www.liander.nl/-/media/files/open-data/historische-15-minuten-bedrijfsmetingen/historische-15-minuten-bedrijfsmetingen.zip"
    weather_data_download_url: ClassVar[str] = "todo"

    @classmethod
    def download(cls, download_dir: Path) -> None:
        # Download and extract both measurements and forecasts
        pass
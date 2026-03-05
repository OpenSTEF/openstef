# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Data-saving callback for forecasting workflows.

Saves intermediate datasets (training data, prepared inputs, forecasts,
contributions) to parquet files. Useful for debugging, backtesting analysis,
and inspecting model behaviour.
"""

import logging
from pathlib import Path
from typing import Any, override

import pandas as pd
from pydantic import Field

from openstef_core.base_model import BaseConfig
from openstef_core.datasets import TimeSeriesDataset, VersionedTimeSeriesDataset
from openstef_core.datasets.validated_datasets import ForecastDataset
from openstef_models.mixins.callbacks import WorkflowContext
from openstef_models.models.forecasting_model import ModelFitResult
from openstef_models.workflows.custom_forecasting_workflow import (
    CustomForecastingWorkflow,
    ForecastingCallback,
)

_logger = logging.getLogger(__name__)


class DataSaveCallback(BaseConfig, ForecastingCallback):
    """Saves intermediate datasets to parquet files during workflow execution.

    Toggle individual outputs via the boolean fields. All paths use
    ``workflow.run_name`` as an identifier in the filename.
    """

    cache_dir: Path = Field(description="Directory to write parquet files to.")
    save_training_data: bool = Field(default=True, description="Save raw training data on fit.")
    save_prepared_data: bool = Field(default=True, description="Save preprocessed training data on fit.")
    save_predict_data: bool = Field(default=True, description="Save prediction input data on predict.")
    save_forecast: bool = Field(default=True, description="Save forecast output on predict.")
    save_contributions: bool = Field(default=False, description="Save prediction contributions on predict.")

    @override
    def model_post_init(self, context: Any) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @override
    def on_fit_start(
        self,
        context: WorkflowContext[CustomForecastingWorkflow],
        data: VersionedTimeSeriesDataset | TimeSeriesDataset,
    ) -> None:
        if self.save_prepared_data:
            # Stash training data so on_fit_end can call prepare_input with it
            context.data["_datasave_training_data"] = data

        if self.save_training_data:
            run_name = context.workflow.run_name or "step"
            data.to_parquet(path=self.cache_dir / f"debug_{run_name}_training.parquet")

    @override
    def on_fit_end(
        self,
        context: WorkflowContext[CustomForecastingWorkflow],
        result: ModelFitResult,
    ) -> None:
        if not self.save_prepared_data:
            return

        training_data = context.data.pop("_datasave_training_data", None)
        if not isinstance(training_data, TimeSeriesDataset):
            return

        run_name = context.workflow.run_name or "step"
        prepared = context.workflow.model.prepare_input(training_data)
        prepared.to_parquet(path=self.cache_dir / f"debug_{run_name}_prepared_training.parquet")

    @override
    def on_predict_end(
        self,
        context: WorkflowContext[CustomForecastingWorkflow],
        data: VersionedTimeSeriesDataset | TimeSeriesDataset,
        result: ForecastDataset,
    ) -> None:
        run_name = context.workflow.run_name or "step"

        if self.save_predict_data:
            data.to_parquet(path=self.cache_dir / f"debug_{run_name}_predict.parquet")

        if self.save_forecast:
            result.to_parquet(path=self.cache_dir / f"debug_{run_name}_forecast.parquet")

        if self.save_contributions and isinstance(data, TimeSeriesDataset):
            try:
                contributions = context.workflow.model.predict_contributions(
                    data,
                    forecast_start=result.forecast_start,
                )
            except NotImplementedError:
                return
            df = pd.concat([contributions.data, result.data.drop(columns=["load"])], axis=1)
            df.to_parquet(path=self.cache_dir / f"contrib_{run_name}_predict.parquet")


__all__ = ["DataSaveCallback"]

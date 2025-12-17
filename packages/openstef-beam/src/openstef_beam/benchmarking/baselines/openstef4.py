# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""OpenSTEF 4.0 forecaster for backtesting pipelines."""

import logging
from collections.abc import Callable
from datetime import timedelta
from functools import partial
from pathlib import Path
from typing import Any, cast, override

import pandas as pd
from pydantic import Field, PrivateAttr
from pydantic_extra_types.coordinate import Coordinate

from openstef_beam.backtesting.backtest_forecaster.mixins import (
    BacktestForecasterConfig,
    BacktestForecasterMixin,
)
from openstef_beam.backtesting.restricted_horizon_timeseries import (
    RestrictedHorizonVersionedTimeSeries,
)
from openstef_beam.benchmarking.benchmark_pipeline import (
    BenchmarkContext,
    BenchmarkTarget,
    ForecasterFactory,
)
from openstef_core.base_model import BaseConfig, BaseModel
from openstef_core.datasets import TimeSeriesDataset
from openstef_core.exceptions import FlatlinerDetectedError, NotFittedError
from openstef_core.types import Q
from openstef_meta.models.ensemble_forecasting_model import EnsembleForecastingModel
from openstef_meta.presets import EnsembleWorkflowConfig, create_ensemble_workflow
from openstef_models.presets import ForecastingWorkflowConfig
from openstef_models.workflows.custom_forecasting_workflow import (
    CustomForecastingWorkflow,
)


class WorkflowCreationContext(BaseConfig):
    """Context information for workflow execution within backtesting."""

    step_name: str | None = Field(
        default=None,
        description="Name of the current backtesting step.",
    )


class OpenSTEF4BacktestForecaster(BaseModel, BacktestForecasterMixin):
    """Forecaster that allows using a ForecastingWorkflow to be used in backtesting, specifically for OpenSTEF4 models.

    A new workflow is created each time fit() is called using the provided workflow_factory,
    ensuring fresh model instances for each training cycle during benchmarking.
    """

    config: BacktestForecasterConfig = Field(
        description="Configuration for the backtest forecaster interface",
    )
    workflow_factory: Callable[[WorkflowCreationContext], CustomForecastingWorkflow] = Field(
        description="Factory function that creates a new CustomForecastingWorkflow instance",
    )
    cache_dir: Path = Field(
        description="Directory to use for caching model artifacts during backtesting",
    )
    debug: bool = Field(
        default=False,
        description="When True, saves intermediate input data for debugging",
    )
    contributions: bool = Field(
        default=False,
        description="When True, saves base Forecaster prediction contributions for ensemble models in cache_dir",
    )

    _workflow: CustomForecastingWorkflow | None = PrivateAttr(default=None)
    _is_flatliner_detected: bool = PrivateAttr(default=False)

    _logger: logging.Logger = PrivateAttr(default=logging.getLogger(__name__))

    @override
    def model_post_init(self, context: Any) -> None:
        if self.debug or self.contributions:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    @override
    def quantiles(self) -> list[Q]:
        # Create a workflow instance if needed to get quantiles
        if self._workflow is None:
            self._workflow = self.workflow_factory(WorkflowCreationContext())
        # Extract quantiles from the workflow's model

        if isinstance(self._workflow.model, EnsembleForecastingModel):
            name = self._workflow.model.forecaster_names[0]
            return self._workflow.model.forecasters[name].config.quantiles
        return self._workflow.model.forecaster.config.quantiles

    @override
    def fit(self, data: RestrictedHorizonVersionedTimeSeries) -> None:
        # Create a new workflow for this training cycle
        context = WorkflowCreationContext(step_name=data.horizon.isoformat())
        workflow = self.workflow_factory(context)

        # Extract the dataset for training
        training_data = data.get_window(
            start=data.horizon - self.config.training_context_length,
            end=data.horizon,
            available_before=data.horizon,
        )

        if self.debug:
            id_str = data.horizon.strftime("%Y%m%d%H%M%S")
            training_data.to_parquet(path=self.cache_dir / f"debug_{id_str}_training.parquet")

        try:
            # Use the workflow's fit method
            workflow.fit(data=training_data)
            self._is_flatliner_detected = False
        except FlatlinerDetectedError:
            self._logger.warning("Flatliner detected during training")
            self._is_flatliner_detected = True
            return  # Skip setting the workflow on flatliner detection

        self._workflow = workflow

        if self.debug:
            id_str = data.horizon.strftime("%Y%m%d%H%M%S")
            self._workflow.model.prepare_input(training_data).to_parquet(  # pyright: ignore[reportPrivateUsage]
                path=self.cache_dir / f"debug_{id_str}_prepared_training.parquet"
            )

    @override
    def predict(self, data: RestrictedHorizonVersionedTimeSeries) -> TimeSeriesDataset | None:
        if self._is_flatliner_detected:
            self._logger.info("Skipping prediction due to prior flatliner detection")
            return None

        if self._workflow is None:
            raise NotFittedError("Must call fit() before predict()")

        # Extract the dataset including both historical context and forecast period
        predict_data = data.get_window(
            start=data.horizon - self.config.predict_context_length,
            end=data.horizon + self.config.predict_length,  # Include the forecast period
            available_before=data.horizon,  # Only use data available at prediction time (prevents lookahead bias)
        )

        try:
            forecast = self._workflow.predict(
                data=predict_data,
                forecast_start=data.horizon,  # Where historical data ends and forecasting begins
            )
        except FlatlinerDetectedError:
            self._logger.info("Flatliner detected during prediction")
            return None

        if self.debug:
            id_str = data.horizon.strftime("%Y%m%d%H%M%S")
            predict_data.to_parquet(path=self.cache_dir / f"debug_{id_str}_predict.parquet")
            forecast.to_parquet(path=self.cache_dir / f"debug_{id_str}_forecast.parquet")

        if self.contributions and isinstance(self._workflow.model, EnsembleForecastingModel):
            contr_str = data.horizon.strftime("%Y%m%d%H%M%S")
            contributions = self._workflow.model.predict_contributions(predict_data, forecast_start=data.horizon)
            df = pd.concat([contributions, forecast.data.drop(columns=["load"])], axis=1)

            df.to_parquet(path=self.cache_dir / f"contrib_{contr_str}_predict.parquet")
        return forecast


class OpenSTEF4PresetBacktestForecaster(OpenSTEF4BacktestForecaster):
    pass


def _preset_target_forecaster_factory(
    base_config: ForecastingWorkflowConfig | EnsembleWorkflowConfig,
    backtest_config: BacktestForecasterConfig,
    cache_dir: Path,
    context: BenchmarkContext,
    target: BenchmarkTarget,
) -> OpenSTEF4BacktestForecaster:
    from openstef_models.presets import create_forecasting_workflow  # noqa: PLC0415
    from openstef_models.presets.forecasting_workflow import LocationConfig  # noqa: PLC0415

    # Factory function that creates a forecaster for a given target.
    prefix = context.run_name

    def _create_workflow(context: WorkflowCreationContext) -> CustomForecastingWorkflow:
        # Create a new workflow instance with fresh model.
        if isinstance(base_config, EnsembleWorkflowConfig):
            return create_ensemble_workflow(
                config=base_config.model_copy(
                    update={
                        "model_id": f"{prefix}_{target.name}",
                        "location": LocationConfig(
                            name=target.name,
                            description=target.description,
                            coordinate=Coordinate(
                                latitude=target.latitude,
                                longitude=target.longitude,
                            ),
                        ),
                    }
                )
            )

        return create_forecasting_workflow(
            config=base_config.model_copy(
                update={
                    "model_id": f"{prefix}_{target.name}",
                    "run_name": context.step_name,
                    "location": LocationConfig(
                        name=target.name,
                        description=target.description,
                        coordinate=Coordinate(
                            latitude=target.latitude,
                            longitude=target.longitude,
                        ),
                    ),
                }
            )
        )

    return OpenSTEF4BacktestForecaster(
        config=backtest_config,
        workflow_factory=_create_workflow,
        debug=False,
        cache_dir=cache_dir / f"{context.run_name}_{target.name}",
    )


def create_openstef4_preset_backtest_forecaster(
    workflow_config: ForecastingWorkflowConfig | EnsembleWorkflowConfig,
    backtest_config: BacktestForecasterConfig | None = None,
    cache_dir: Path = Path("cache"),
) -> ForecasterFactory[BenchmarkTarget]:
    """Create a factory that returns an OpenSTEF4BacktestForecaster for a benchmark target.

    Args:
        workflow_config: The configured `ForecastingWorkflowConfig` that will be cloned and
            assigned to a target-specific workflow instance.
        backtest_config: Optional `BacktestForecasterConfig` to control training/prediction windows.
            If None, a sensible default is created.
        cache_dir: Directory to store cached artifacts for created forecasters. A subdirectory will be
            created per benchmark run and target.

    Returns:
        A `ForecasterFactory[BenchmarkTarget]` partial which accepts a `BenchmarkContext` and a
        `BenchmarkTarget` and returns a configured `OpenSTEF4BacktestForecaster`.
    """
    if backtest_config is None:
        backtest_config = BacktestForecasterConfig(
            requires_training=True,
            predict_length=timedelta(days=7),
            predict_min_length=timedelta(minutes=15),
            predict_context_length=timedelta(days=14),  # Context needed for lag features
            predict_context_min_coverage=0.5,
            training_context_length=timedelta(days=90),  # Three months of training data
            training_context_min_coverage=0.5,
            predict_sample_interval=timedelta(minutes=15),
        )

    return cast(
        ForecasterFactory[BenchmarkTarget],
        partial(
            _preset_target_forecaster_factory,
            workflow_config,
            backtest_config,
            cache_dir,
        ),
    )


__all__ = [
    "OpenSTEF4BacktestForecaster",
    "WorkflowCreationContext",
    "create_openstef4_preset_backtest_forecaster",
]

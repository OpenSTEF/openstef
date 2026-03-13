# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""OpenSTEF 4.0 forecaster for backtesting pipelines.

Requires the ``baselines`` extra: ``pip install openstef-beam[baselines]``.
"""

import logging
from datetime import timedelta
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast, override

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
from openstef_core.base_model import BaseModel
from openstef_core.datasets import TimeSeriesDataset
from openstef_core.exceptions import (
    FlatlinerDetectedError,
    InsufficientlyCompleteError,
    MissingExtraError,
)
from openstef_core.types import Q
from openstef_models.presets import ForecastingWorkflowConfig, create_forecasting_workflow
from openstef_models.presets.forecasting_workflow import LocationConfig
from openstef_models.workflows.callbacks.data_save import DataSaveCallback
from openstef_models.workflows.custom_forecasting_workflow import (
    CustomForecastingWorkflow,
    ForecastingCallback,
)

if TYPE_CHECKING:
    from openstef_meta.presets import EnsembleForecastingWorkflowConfig


class OpenSTEF4BacktestForecaster(BaseModel, BacktestForecasterMixin):
    """Forecaster that allows using a ForecastingWorkflow to be used in backtesting, specifically for OpenSTEF4 models.

    A new workflow is created each time fit() is called using the provided workflow_factory,
    ensuring fresh model instances for each training cycle during benchmarking.
    """

    config: BacktestForecasterConfig = Field(
        description="Configuration for the backtest forecaster interface",
    )
    workflow_template: CustomForecastingWorkflow = Field(
        description="Untrained workflow template; deep-copied for each fit() call",
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
        description="When True, saves base forecaster prediction contributions for ensemble models",
    )
    extra_callbacks: list[ForecastingCallback] = Field(
        default_factory=list[ForecastingCallback],
        description="Additional callbacks to inject into workflows created by the factory.",
    )

    _workflow: CustomForecastingWorkflow | None = PrivateAttr(default=None)
    _is_flatliner_detected: bool = PrivateAttr(default=False)

    _logger: logging.Logger = PrivateAttr(default=logging.getLogger(__name__))

    @override
    def model_post_init(self, context: Any) -> None:
        if self.debug or self.contributions:
            self.extra_callbacks.append(
                DataSaveCallback(
                    cache_dir=self.cache_dir,
                    save_training_data=self.debug,
                    save_prepared_data=self.debug,
                    save_predict_data=self.debug,
                    save_forecast=self.debug,
                    save_contributions=self.contributions,
                )
            )

    @property
    @override
    def quantiles(self) -> list[Q]:
        return self.workflow_template.model.quantiles

    @override
    def fit(self, data: RestrictedHorizonVersionedTimeSeries) -> None:
        # Deep-copy the template for a fresh model
        workflow = self.workflow_template.with_run_name(data.horizon.isoformat())
        workflow.callbacks.extend(self.extra_callbacks)

        # Extract the dataset for training
        training_data = data.get_window(
            start=data.horizon - self.config.training_context_length,
            end=data.horizon,
            available_before=data.horizon,
        )

        try:
            # Use the workflow's fit method
            workflow.fit(data=training_data)
            self._is_flatliner_detected = False
        except FlatlinerDetectedError:
            self._logger.warning("Flatliner detected during training")
            self._is_flatliner_detected = True
            return  # Skip setting the workflow on flatliner detection
        except InsufficientlyCompleteError:
            self._logger.warning("Insufficient training data at %s, retaining previous model", data.horizon)
            return  # Retain previous model state; predictions will use the last successful fit

        self._workflow = workflow

    @override
    def predict(self, data: RestrictedHorizonVersionedTimeSeries) -> TimeSeriesDataset | None:
        if self._is_flatliner_detected:
            self._logger.info("Skipping prediction due to prior flatliner detection")
            return None

        if self._workflow is None:
            self._logger.info("No fitted model available, skipping prediction")
            return None

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

        return forecast


def _preset_target_forecaster_factory(
    base_config: "ForecastingWorkflowConfig | EnsembleForecastingWorkflowConfig",
    backtest_config: BacktestForecasterConfig,
    cache_dir: Path,
    context: BenchmarkContext,
    target: BenchmarkTarget,
) -> OpenSTEF4BacktestForecaster:
    location = LocationConfig(
        name=target.name,
        description=target.description,
        coordinate=Coordinate(
            latitude=target.latitude,
            longitude=target.longitude,
        ),
    )

    update: dict[str, Any] = {
        "model_id": f"{context.run_name}_{target.name}",
        "location": location,
    }

    if base_config.kind == "ensemble":
        try:
            from openstef_meta.presets import create_ensemble_forecasting_workflow  # noqa: PLC0415
        except ImportError as e:
            raise MissingExtraError("openstef-meta") from e
        workflow = create_ensemble_forecasting_workflow(config=base_config.model_copy(update=update))
    else:
        workflow = create_forecasting_workflow(config=base_config.model_copy(update=update))

    return OpenSTEF4BacktestForecaster(
        config=backtest_config,
        workflow_template=workflow,
        debug=False,
        cache_dir=cache_dir / f"{context.run_name}_{target.name}",
    )


def create_openstef4_preset_backtest_forecaster(
    workflow_config: "ForecastingWorkflowConfig | EnsembleForecastingWorkflowConfig",
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
    "create_openstef4_preset_backtest_forecaster",
]

# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Backtesting pipeline for evaluating energy forecasting models.

Simulates realistic forecasting scenarios by replaying historical data with
proper temporal constraints. Executes prediction and retraining schedules
that mirror operational deployment conditions, ensuring evaluation results
accurately reflect real-world model performance.
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import optuna
import pandas as pd
from optuna.study import Study
from optuna.trial import FrozenTrial, Trial
from pydantic import Field

from openstef_beam.backtesting.backtest_forecaster.mixins import (
    BacktestForecasterConfig,
)
from openstef_beam.backtesting.backtest_forecaster.openstef4_backtest_forecaster import (
    OpenSTEF4BacktestForecaster,
)
from openstef_beam.backtesting.backtest_pipeline import BacktestConfig, BacktestPipeline
from openstef_beam.parameter_tuning.greedy_backtester import (
    GreedyBacktestConfig,
    GreedyBackTestPipeline,
)
from openstef_beam.parameter_tuning.models import (
    CategoricalDistribution,
    FloatDistribution,
    IntDistribution,
    OptimizationMetric,
    ParameterSpace,
)
from openstef_core.base_model import BaseConfig
from openstef_core.datasets import ForecastDataset, ForecastInputDataset
from openstef_core.datasets.versioned_timeseries_dataset import TimeSeriesDataset, VersionedTimeSeriesDataset
from openstef_core.types import AvailableAt, LeadTime, Quantile
from openstef_models.models.forecasting.forecaster import HyperParams
from openstef_models.models.forecasting_model import ForecastingModel
from openstef_models.workflows.custom_forecasting_workflow import (
    CustomForecastingWorkflow,
)

logger = logging.getLogger(__name__)


class BaseOptimizerConfig(BaseConfig):
    """Configuration for the optimizer."""

    parameter_space: ParameterSpace = Field(description="Parameter search space for optimization.")

    quantiles: list[Quantile] = Field(description="Quantiles to be predicted by the forecaster.")

    horizon: LeadTime = Field(description="Forecasting horizon for which the optimization is performed.")

    forecasting_model: ForecastingModel = Field(
        description="The base forecasting model to optimize. Note only the forecaster class is used from this model."
    )

    optimization_metric: OptimizationMetric = Field(description="Metric used for optimization during parameter tuning.")


class RigorousOptimizerConfig(BaseOptimizerConfig):
    """Configuration for the rigorous optimizer."""

    backtest_config: BacktestConfig = Field(description="Backtesting configuration for the optimizer.")

    backtest_forecaster_config: BacktestForecasterConfig = Field(
        description="Backtest forecaster configuration for the optimizer."
    )


class GreedyOptimizerConfig(BaseOptimizerConfig):
    """Configuration for the greedy optimizer."""

    backtest_config: GreedyBacktestConfig = Field(description="Backtesting configuration for the optimizer.")


class BaseOptunaOptimizer(ABC):
    """Optimizer using Optuna for hyperparameter tuning."""

    def __init__(self, config: BaseOptimizerConfig):
        """Initialize the Optuna optimizer.

        Args:
            config: Configuration for the optimizer.
        """
        self._forecaster_class = config.forecasting_model.forecaster.__class__
        self._forecaster_config = config.forecasting_model.forecaster.config
        self._default_hyperparams = config.forecasting_model.forecaster.hyperparams

        self.quantiles = config.quantiles
        self.horizon = config.horizon

        # Validate parameter space
        self.parameter_space: ParameterSpace = config.parameter_space

        self.direction = "minimize" if config.optimization_metric.direction_minimize else "maximize"
        self.metric = config.optimization_metric.metric
        self.metric_name: str = config.optimization_metric.name

        self._base_model = config.forecasting_model

        self.available_at = AvailableAt.from_string("D-1T06:00")

    def _make_forecasting_workflow(
        self, hyperparams: HyperParams, model_id: int | None = None
    ) -> CustomForecastingWorkflow:
        model = self._make_forecasting_model(hyperparams=hyperparams)

        return CustomForecastingWorkflow(
            model=model,
            model_id=f"optuna_workflow_{model_id}" if model_id is not None else "optuna_workflow",
            callbacks=[],
        )

    def _make_forecasting_model(self, hyperparams: HyperParams) -> ForecastingModel:
        forecaster_config = self._forecaster_config.model_copy(
            update={
                "quantiles": self.quantiles,
                "horizons": [self.horizon],
                "hyperparams": hyperparams,
            }
        )
        forecaster = self._forecaster_class(config=forecaster_config)
        return self._base_model.model_copy(update={"forecaster": forecaster})

    def _make_optuna_space(self, trial: Trial) -> dict[str, Any]:
        optuna_space: dict[str, str | float | int | None] = {}
        for param_name, distribution in self.parameter_space.items():
            if isinstance(distribution, FloatDistribution):
                optuna_space[param_name] = trial.suggest_float(
                    param_name,
                    distribution.low,
                    distribution.high,
                    log=distribution.log,
                    step=distribution.step,
                )
            elif isinstance(distribution, IntDistribution):
                optuna_space[param_name] = trial.suggest_int(
                    param_name,
                    distribution.low,
                    distribution.high,
                    log=distribution.log,
                    step=distribution.step,
                )
            elif isinstance(distribution, CategoricalDistribution):
                optuna_space[param_name] = trial.suggest_categorical(param_name, distribution.choices)
            else:
                message = f"Unsupported distribution type: {type(distribution)}"
                raise TypeError(message)
        return optuna_space

    def _score_predictions(
        self, predictions: ForecastDataset | TimeSeriesDataset, ground_truth: TimeSeriesDataset
    ) -> float:
        """Score the predictions using the a pre-set optimization metric.

        Args:
        predictions: The predictions made by the forecasting model.
        ground_truth: The ground truth dataset.

        Returns:
        The metric value as a float.
        """
        target = ForecastInputDataset.from_timeseries(
            dataset=ground_truth,
            target_column="load",
        )
        target = target.pipe_pandas(pd.DataFrame.dropna)  # type: ignore

        predictions_filtered = predictions.filter_by_available_at(available_at=self.available_at).select_version()

        # Remove target column from predictions to avoid duplication
        if "load" in predictions_filtered.data.columns:
            predictions_filtered = predictions_filtered.pipe_pandas(lambda df: df.drop(columns=["load"]))

        final_set = ForecastDataset(
            data=target.data.join(predictions_filtered.data, how="inner"),
            sample_interval=predictions.sample_interval,
            target_column="load",
        ).filter_quantiles(quantiles=self.quantiles)

        metric_value = self.metric(final_set)

        if "global" in metric_value:
            metric_value = metric_value["global"]

        return float(metric_value[self.metric_name])  # type: ignore

    @abstractmethod
    def _make_backtest(self, hyperparams: HyperParams) -> BacktestPipeline | GreedyBackTestPipeline:
        """Create a backtest pipeline for the given hyperparameters."""
        raise NotImplementedError("Subclasses must implement _make_backtest")

    def optimize(
        self,
        predictors: VersionedTimeSeriesDataset,
        ground_truth: VersionedTimeSeriesDataset,
    ) -> HyperParams:
        """Optimize hyperparameters using Optuna.

        Args:
            predictors: The predictors dataset.
            ground_truth: The ground truth dataset.

        Returns:
            The best hyperparameters found during optimization.
        """
        ground_truth_selected = ground_truth.select_version()

        def objective(trial: Trial) -> float:
            # Convert to Optuna
            params = self._make_optuna_space(trial)
            # Generate HyperParams and Config
            hyperparams = self._default_hyperparams.model_copy(update=params)

            backtest = self._make_backtest(hyperparams=hyperparams)

            predictions: ForecastDataset | TimeSeriesDataset = backtest.run(
                predictors=predictors,
                ground_truth=ground_truth,
                start=ground_truth.index[0],
                end=ground_truth.index[-1],
            )

            return self._score_predictions(predictions=predictions, ground_truth=ground_truth_selected)

        def logger_callback(study: Study, trial: FrozenTrial) -> None:
            logger.info("Current value: %s, Current params: %s", trial.value, trial.params)
            logger.info(
                "Best value: %s, Best params: %s",
                study.best_value,
                study.best_trial.params,
            )

        study = optuna.create_study(direction=self.direction)

        study.optimize(objective, timeout=3600, n_jobs=1, n_trials=100, callbacks=[logger_callback])

        return self._default_hyperparams.model_copy(update=study.best_trial.params)


class RigorousOptunaOptimizer(BaseOptunaOptimizer):
    """Optimizer using Optuna for hyperparameter tuning with rigorous backtesting."""

    def __init__(self, config: RigorousOptimizerConfig):
        """Initialize the rigorous Optuna optimizer."""
        self.backtest_config = config.backtest_config
        self.backtest_forecaster_config = config.backtest_forecaster_config
        super().__init__(config=config)

    def _make_backtest(self, hyperparams: HyperParams) -> BacktestPipeline:
        forecaster = self._make_backtest_forecaster(hyperparams=hyperparams)

        return BacktestPipeline(
            forecaster=forecaster,
            config=self.backtest_config,
        )

    def _make_backtest_forecaster(self, hyperparams: HyperParams) -> OpenSTEF4BacktestForecaster:
        def _make_workflow() -> CustomForecastingWorkflow:
            return self._make_forecasting_workflow(hyperparams=hyperparams, model_id=None)

        return OpenSTEF4BacktestForecaster(
            config=self.backtest_forecaster_config,
            workflow_factory=_make_workflow,
            cache_dir=Path("/tmp/optuna_backtest_cache"),
            debug=False,
        )


class GreedyOptunaOptimizer(BaseOptunaOptimizer):
    """Optimizer using Optuna for hyperparameter tuning with greedy backtesting."""

    def __init__(self, config: GreedyOptimizerConfig):
        """Initialize the greedy Optuna optimizer."""
        super().__init__(config=config)
        self.backtest_config = config.backtest_config

    def _make_backtest(self, hyperparams: HyperParams) -> GreedyBackTestPipeline:
        backtest_config = self.backtest_config.model_copy(
            update={"forecasting_model": self._make_forecasting_model(hyperparams=hyperparams)}
        )

        return GreedyBackTestPipeline(config=backtest_config)

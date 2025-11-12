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
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any, Literal

import optuna
import pandas as pd
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
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
from openstef_beam.benchmarking.benchmarks.liander2024 import Liander2024TargetProvider
from openstef_beam.parameter_tuning.greedy_backtester import (
    GreedyBacktestConfig,
    GreedyBackTestPipeline,
)
from openstef_beam.parameter_tuning.models import (
    OptimizationMetric,
    ParameterSpace,
)
from openstef_core.base_model import BaseConfig
from openstef_core.datasets import ForecastDataset, ForecastInputDataset
from openstef_core.datasets.versioned_timeseries_dataset import TimeSeriesDataset, VersionedTimeSeriesDataset
from openstef_core.types import LeadTime, Quantile
from openstef_core.utils.multiprocessing import run_parallel
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

    n_jobs: int = Field(default=1, description="Number of parallel jobs to run during optimization.")
    n_trials: int = Field(default=100, description="Number of trials to run during optimization.")
    timeout: int = Field(default=3600, description="Timeout in seconds for the optimization process.")
    verbosity: Literal["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"] = Field(
        default="INFO", description="Verbosity level for the optimizer."
    )


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

        self.n_trials = config.n_trials
        self.n_jobs = config.n_jobs

        logger.setLevel(config.verbosity)

        # Parameter space
        self.parameter_space: ParameterSpace = config.parameter_space

        self.direction = "minimize" if config.optimization_metric.direction_minimize else "maximize"
        self.metric = config.optimization_metric.metric
        self.metric_name: str = config.optimization_metric.name

        self._base_model = config.forecasting_model

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

        predictions_selected = predictions.select_version()

        # Remove target column from predictions to avoid duplication
        if "load" in predictions_selected.data.columns:
            predictions_selected = predictions_selected.pipe_pandas(lambda df: df.drop(columns=["load"]))

        final_set = ForecastDataset(
            data=target.data.join(predictions_selected.data, how="inner"),
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

    @staticmethod
    def logger_callback(study: Study, trial: FrozenTrial) -> None:
        """Log the progress of the optimization study. To be used as an Optuna callback."""
        logger.info("Current value: %s, Current params: %s", trial.value, trial.params)
        logger.info(
            "Best value: %s, Best params: %s",
            study.best_value,
            study.best_trial.params,
        )

    @staticmethod
    def objective_target_provider(
        trial: Trial,
        parameter_space: ParameterSpace,
        default_params: HyperParams,
        backtest_maker: Callable[[HyperParams], BacktestPipeline | GreedyBackTestPipeline],
        scoring_function: Callable[[ForecastDataset | TimeSeriesDataset, TimeSeriesDataset], float],
    ) -> float:
        """Objective function for optimizing over multiple targets.

        Args:
            trial: The Optuna trial object.
            parameter_space: The parameter space to sample from.
            default_params: The default hyperparameters of the model.
            backtest_maker: Function to create a backtest pipeline given hyperparameters.
            scoring_function: Function to score the predictions against ground truth.

        Returns:
            The average metric score across all targets.
        """
        params = parameter_space.make_optuna(trial)
        hyperparams = default_params.model_copy(update=params)
        target_provider = Liander2024TargetProvider(data_dir=Path("../data/liander2024-energy-forecasting-benchmark"))

        metrics: list[float] = []
        for target in target_provider.get_targets():
            predictors = target_provider.get_predictors_for_target(target)
            ground_truth = target_provider.get_measurements_for_target(target)

            backtest = backtest_maker(hyperparams)

            predictions: ForecastDataset | TimeSeriesDataset = backtest.run(
                predictors=predictors,
                ground_truth=ground_truth,
                start=ground_truth.index[0],
                end=ground_truth.index[-1],
            )
            score = scoring_function(predictions, ground_truth.select_version())
            metrics.append(score)

        return sum(metrics) / len(metrics)

    def optimize_target_provider(self) -> HyperParams:
        """Optimize hyperparameters using Optuna over multiple targets.

        Args:
            target_provider: The target provider to get targets, predictors and ground truth from.

        Returns:
            The best hyperparameters found during optimization.
        """
        objective = partial(
            self.objective_target_provider,
            parameter_space=self.parameter_space,
            default_params=self._default_hyperparams,
            backtest_maker=self._make_backtest,
            scoring_function=self._score_predictions,
        )

        return self._run_optimization(objective=objective, n_trials=self.n_trials, n_jobs=self.n_jobs)

    @staticmethod
    def objective_dataset(
        trial: Trial,
        parameter_space: ParameterSpace,
        default_params: HyperParams,
        backtest_maker: Callable[[HyperParams], BacktestPipeline | GreedyBackTestPipeline],
        scoring_function: Callable[[ForecastDataset | TimeSeriesDataset, TimeSeriesDataset], float],
        predictors: VersionedTimeSeriesDataset,
        ground_truth: VersionedTimeSeriesDataset,
    ) -> float:
        """Objective function for optimizing over a single dataset."""
        # Convert to Optuna
        params = parameter_space.make_optuna(trial)
        # Generate HyperParams and Config
        hyperparams = default_params.model_copy(update=params)

        backtest = backtest_maker(hyperparams)

        predictions: ForecastDataset | TimeSeriesDataset = backtest.run(
            predictors=predictors,
            ground_truth=ground_truth,
            start=ground_truth.index[0],
            end=ground_truth.index[-1],
        )

        return scoring_function(predictions, ground_truth.select_version())

    def optimize_dataset(
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
        objective = partial(
            self.objective_dataset,
            parameter_space=self.parameter_space,
            default_params=self._default_hyperparams,
            backtest_maker=self._make_backtest,
            scoring_function=self._score_predictions,
            predictors=predictors,
            ground_truth=ground_truth,
        )

        return self._run_optimization(objective=objective, n_trials=self.n_trials, n_jobs=self.n_jobs)

    @staticmethod
    def _run_optimization_job(
        _: int, config: dict[str, Any], objective: Callable[..., float], logger_callback: Callable[..., None]
    ) -> None:
        study = optuna.create_study(
            study_name="journal_storage_multiprocess",
            storage=JournalStorage(JournalFileBackend(file_path="./journal.log")),
            load_if_exists=True,
            direction=config["direction"],
        )
        study.optimize(
            objective,
            timeout=3600,
            n_jobs=config["n_jobs"],
            n_trials=config["trials_per_job"],
            callbacks=[logger_callback],
        )

    def _run_optimization(self, objective: Callable[..., float], n_trials: int, n_jobs: int) -> HyperParams:
        if n_jobs > 1:
            trials_per_job = int(n_trials / n_jobs) + 1
            journal_path = Path("./journal.log")
            if journal_path.exists():
                journal_path.unlink()

            run_parallel(
                partial(
                    self._run_optimization_job,
                    config={
                        "direction": self.direction,
                        "n_jobs": 1,
                        "trials_per_job": trials_per_job,
                    },
                    objective=objective,
                    logger_callback=self.logger_callback,
                ),
                items=iter(range(trials_per_job)),
                n_processes=n_jobs,
            )
            study = optuna.load_study(
                study_name="journal_storage_multiprocess",
                storage=JournalStorage(JournalFileBackend(file_path="./journal.log")),
            )

        else:
            study = optuna.create_study(direction=self.direction)
            study.optimize(objective, timeout=3600, n_jobs=1, n_trials=n_trials, callbacks=[self.logger_callback])

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

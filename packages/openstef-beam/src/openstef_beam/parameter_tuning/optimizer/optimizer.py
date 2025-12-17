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
from optuna.trial import FrozenTrial
from pydantic import Field

from openstef_beam.backtesting.backtest_forecaster.mixins import (
    BacktestForecasterConfig,
)
from openstef_beam.backtesting.backtest_pipeline import BacktestConfig, BacktestPipeline
from openstef_beam.benchmarking.baselines.openstef4 import (
    OpenSTEF4BacktestForecaster,
    WorkflowCreationContext,
)
from openstef_beam.benchmarking.benchmarks.liander2024 import Liander2024TargetProvider
from openstef_beam.benchmarking.target_provider import BenchmarkTarget
from openstef_beam.parameter_tuning.models import (
    OptimizationMetric,
    ParameterSpace,
)
from openstef_core.base_model import BaseConfig
from openstef_core.datasets import ForecastDataset, ForecastInputDataset
from openstef_core.datasets.versioned_timeseries_dataset import TimeSeriesDataset
from openstef_core.utils.multiprocessing import run_parallel
from openstef_models.models.forecasting.forecaster import HyperParams
from openstef_models.presets import create_forecasting_workflow
from openstef_models.presets.forecasting_workflow import (
    Coordinate,
    ForecastingWorkflowConfig,
    LocationConfig,
)
from openstef_models.workflows.custom_forecasting_workflow import (
    CustomForecastingWorkflow,
)

logger = logging.getLogger(__name__)


class OptimizerConfig(BaseConfig):
    """Configuration for the optimizer."""

    parameter_space: ParameterSpace = Field(description="Parameter search space for optimization.")

    optimization_metric: OptimizationMetric = Field(description="Metric used for optimization during parameter tuning.")

    n_jobs: int = Field(default=1, description="Number of parallel jobs to run during optimization.")
    n_trials: int = Field(default=100, description="Number of trials to run during optimization.")
    timeout: int = Field(default=3600, description="Timeout in seconds for the optimization process.")
    verbosity: Literal["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"] = Field(
        default="INFO", description="Verbosity level for the optimizer."
    )

    backtest_config: BacktestConfig = Field(description="Backtesting configuration for the optimizer.")

    backtest_forecaster_config: BacktestForecasterConfig = Field(
        description="Backtest forecaster configuration for the optimizer."
    )

    base_config: ForecastingWorkflowConfig = Field(
        description="Base configuration for creating forecasting workflows during backtesting."
    )

    cache_dir: Path = Field(
        default=Path("./optimizer_results/cache"),
        description="Directory to use for caching model artifacts during backtesting",
    )


class OptimizerRunContext(BaseConfig):
    """Context information for workflow execution within backtesting."""

    base_config: ForecastingWorkflowConfig = Field(
        description="Base configuration for creating forecasting workflows during backtesting."
    )

    parameter_space: ParameterSpace = Field(description="ParameterSpace for the optimization run")

    target_provider: Liander2024TargetProvider = Field(description="Target provider")


class OptimizerTrialContext(BaseConfig):
    """Context information for workflow execution within backtesting."""

    base_config: ForecastingWorkflowConfig = Field(
        description="Base configuration for creating forecasting workflows during backtesting."
    )

    hyperparams: HyperParams = Field(description="Hyperparameters for the current optimization trial.")

    target: BenchmarkTarget = Field(description="Benchmark target for the current optimization trial.")


class OptunaOptimizer(ABC):
    """Optimizer using Optuna for hyperparameter tuning."""

    def __init__(self, config: OptimizerConfig):
        """Initialize the Optuna optimizer.

        Args:
            config: Configuration for the optimizer.
        """
        self.base_config = config.base_config
        self.backtest_config = config.backtest_config
        self.backtest_forecaster_config = config.backtest_forecaster_config

        self.n_trials = config.n_trials
        self.n_jobs = config.n_jobs

        self.cache_dir = config.cache_dir

        self.horizons = config.base_config.horizons
        self.quantiles = config.base_config.quantiles

        logger.setLevel(config.verbosity)

        # Parameter space
        self.parameter_space: ParameterSpace = config.parameter_space

        self.direction = "minimize" if config.optimization_metric.direction_minimize else "maximize"
        self.metric = config.optimization_metric.metric
        self.metric_name: str = config.optimization_metric.name

    @abstractmethod
    def optimize(self, *args: Any, **kwargs: Any) -> HyperParams:
        """Optimize hyperparameters using Optuna.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            The best hyperparameters found during optimization.
        """
        msg = "Subclasses must implement the optimize method."
        raise NotImplementedError(msg)

    @abstractmethod
    def objective(self, *args: Any, **kwargs: Any) -> float:
        """Objective function for optimizing hyperparameters.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            The metric value as a float.
        """
        msg = "Subclasses must implement the objective method."
        raise NotImplementedError(msg)

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
    def _run_optimization_job(
        _: int, config: dict[str, Any], objective: Callable[..., float], logger_callback: Callable[..., None]
    ) -> None:
        study = optuna.create_study(
            study_name=config["experiment_name"],
            storage=JournalStorage(JournalFileBackend(file_path=config["log_path"])),
            load_if_exists=True,
            direction=config["direction"],
        )
        study.optimize(
            objective,
            timeout=3600,
            n_jobs=1,
            n_trials=config["trials_per_job"],
            callbacks=[logger_callback],
        )

    def _run_optimization_parallel(
        self, experiment_name: str, objective: Callable[..., float], n_trials: int, n_jobs: int
    ) -> HyperParams:
        path = f"./optimization_results/{experiment_name}_journal.log"
        # TODO: Clean up existing journal file if exists
        journal_path = Path(path)
        if journal_path.exists():
            journal_path.unlink()

        trials_per_job = int(n_trials / n_jobs) + 1

        run_parallel(
            partial(
                self._run_optimization_job,
                config={
                    "experiment_name": experiment_name,
                    "direction": self.direction,
                    "trials_per_job": trials_per_job,
                    "log_path": path,
                },
                objective=objective,
                logger_callback=self.logger_callback,
            ),
            items=iter(range(trials_per_job)),
            n_processes=n_jobs,
        )
        study = optuna.load_study(
            study_name=experiment_name,
            storage=JournalStorage(JournalFileBackend(file_path=path)),
        )

        return self.parameter_space.default_hyperparams().model_copy(update=study.best_trial.params)

    def _run_optimization_sequential(
        self, experiment_name: str, objective: Callable[..., float], n_trials: int
    ) -> HyperParams:
        path = f"./optimization_results/{experiment_name}_journal.log"
        # TODO: Clean up existing journal file if exists
        journal_path = Path(path)
        if journal_path.exists():
            journal_path.unlink()

        study = optuna.create_study(
            study_name=experiment_name,
            direction=self.direction,
            storage=JournalStorage(JournalFileBackend(file_path=path)),
        )
        study.optimize(objective, timeout=3600, n_trials=n_trials, callbacks=[self.logger_callback])

        return self.parameter_space.default_hyperparams().model_copy(update=study.best_trial.params)

    def _make_backtest(self, context: OptimizerTrialContext) -> BacktestPipeline:
        forecaster = self._make_backtest_forecaster(context=context)

        return BacktestPipeline(
            forecaster=forecaster,
            config=self.backtest_config,
        )

    def _make_backtest_forecaster(self, context: OptimizerTrialContext) -> OpenSTEF4BacktestForecaster:

        prefix = "test"
        target = context.target
        base_config = context.base_config
        hyperparams = context.hyperparams

        def _create_workflow(context: WorkflowCreationContext) -> CustomForecastingWorkflow:
            # Create a new workflow instance with fresh model.
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
                        f"{base_config.model}_hyperparams": hyperparams,
                    }
                )
            )

        return OpenSTEF4BacktestForecaster(
            config=self.backtest_forecaster_config,
            workflow_factory=_create_workflow,
            cache_dir=self.cache_dir,
            debug=False,
        )

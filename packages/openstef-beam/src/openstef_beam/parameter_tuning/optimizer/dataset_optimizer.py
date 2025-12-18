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
from collections.abc import Callable
from functools import partial

from optuna.trial import Trial

from openstef_beam.backtesting.backtest_pipeline import BacktestPipeline
from openstef_beam.benchmarking.target_provider import BenchmarkTarget
from openstef_beam.parameter_tuning.optimizer.optimizer import OptimizerTrialContext, OptunaOptimizer
from openstef_core.datasets.versioned_timeseries_dataset import TimeSeriesDataset, VersionedTimeSeriesDataset
from openstef_models.models.forecasting.forecaster import HyperParams
from openstef_models.presets.forecasting_workflow import (
    Latitude,
    Longitude,
)

logger = logging.getLogger(__name__)


class DatasetOptimizer(OptunaOptimizer):
    """Optuna optimizer aimed at optimizing individual datasets."""

    def optimize(
        self,
        predictors: VersionedTimeSeriesDataset,
        ground_truth: VersionedTimeSeriesDataset,
        experiment_name: str,
    ) -> HyperParams:
        """Optimize hyperparameters using Optuna.

        Args:
            predictors: The predictors dataset.
            ground_truth: The ground truth dataset.
            experiment_name: Name of the optimization experiment.

        Returns:
            The best hyperparameters found during optimization.
        """
        objective = partial(
            self.objective,
            backtest_maker=self._make_backtest,
            scoring_function=self._score_predictions,
            predictors=predictors,
            ground_truth=ground_truth,
        )

        if self.n_jobs > 1:
            return self._run_optimization_parallel(
                experiment_name=experiment_name, objective=objective, n_trials=self.n_trials, n_jobs=self.n_jobs
            )

        return self._run_optimization_sequential(
            objective=objective,
            experiment_name=experiment_name,
            n_trials=self.n_trials,
        )

    def objective(
        self,
        trial: Trial,
        backtest_maker: Callable[[OptimizerTrialContext], BacktestPipeline],
        scoring_function: Callable[[TimeSeriesDataset, TimeSeriesDataset], float],
        predictors: VersionedTimeSeriesDataset,
        ground_truth: VersionedTimeSeriesDataset,
    ) -> float:
        """Objective function for optimizing over a single dataset.

        Args:
            trial: The Optuna trial object.
            backtest_maker: Function to create a backtest pipeline for given hyperparameters.
            scoring_function: Function to score predictions against ground truth.
            predictors: The predictors dataset.
            ground_truth: The ground truth dataset.

        Returns:
            The score achieved with the given hyperparameters.
        """
        # Convert to Optuna
        params = self.parameter_space.make_optuna(trial)

        # Generate HyperParams
        hyperparams = self.parameter_space.default_hyperparams().model_copy(update=params)

        trial_context = OptimizerTrialContext(
            base_config=self.base_config,
            hyperparams=hyperparams,
            target=BenchmarkTarget(
                name="single_dataset",
                description="Dataset provided directly for optimization",
                latitude=Latitude(0.0),
                longitude=Longitude(0.0),
                benchmark_start=ground_truth.index[0],
                benchmark_end=ground_truth.index[-1],
                train_start=ground_truth.index[0],
            ),
        )

        backtest = backtest_maker(trial_context)

        predictions: TimeSeriesDataset = backtest.run(
            predictors=predictors,
            ground_truth=ground_truth,
            start=ground_truth.index[0],
            end=ground_truth.index[-1],
        )

        return scoring_function(predictions, ground_truth.select_version())

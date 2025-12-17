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
from openstef_beam.benchmarking.benchmarks.liander2024 import Liander2024TargetProvider
from openstef_beam.benchmarking.target_provider import BenchmarkTarget
from openstef_beam.parameter_tuning.optimizer.optimizer import OptimizerTrialContext, OptunaOptimizer
from openstef_core.datasets.versioned_timeseries_dataset import TimeSeriesDataset
from openstef_core.utils.multiprocessing import run_parallel
from openstef_models.models.forecasting.forecaster import HyperParams
from openstef_models.presets.forecasting_workflow import (
    ForecastingWorkflowConfig,
)

logger = logging.getLogger(__name__)


class BenchmarkOptimizer(OptunaOptimizer):
    """Optuna optimizer for optimizing multiple targets via the Benchmark Interface."""

    def objective(
        self,
        trial: Trial,
        target_provider: Liander2024TargetProvider,
        score_for_target: Callable[[HyperParams, BenchmarkTarget], float],
        n_jobs: int = 1,
    ) -> float:
        """Objective function for optimizing over multiple targets.

        Args:
            trial: The Optuna trial object.
            target_provider: The target provider to get targets, predictors and ground truth from.
            score_for_target: Function to score a single target with given hyperparameters.
            n_jobs: Number of parallel jobs to run.

        Returns:
            The average metric score across all targets.
        """
        # Draw hyperparameters from the parameter space
        hyperparams = self.parameter_space.default_hyperparams().model_copy(
            update=self.parameter_space.make_optuna(trial)
        )

        metrics: list[float] = []

        if n_jobs > 1:
            score_with_params = partial(score_for_target, hyperparams)

            metrics = run_parallel(
                score_with_params,
                items=target_provider.get_targets()[:4],
                n_processes=n_jobs,
            )
        else:
            for target in target_provider.get_targets():
                score = score_for_target(hyperparams, target)
                metrics.append(score)

        return sum(metrics) / len(metrics)

    @staticmethod
    def _run_trial_for_target(
        hyperparams: HyperParams,
        target: BenchmarkTarget,
        target_provider: Liander2024TargetProvider,
        base_config: ForecastingWorkflowConfig,
        backtest_maker: Callable[[OptimizerTrialContext], BacktestPipeline],
        scoring_maker: Callable[[TimeSeriesDataset, TimeSeriesDataset], float],
    ) -> float:
        predictors = target_provider.get_predictors_for_target(target)
        ground_truth = target_provider.get_measurements_for_target(target)

        trial_context = OptimizerTrialContext(
            base_config=base_config,
            hyperparams=hyperparams,
            target=target,
        )

        backtest = backtest_maker(trial_context)

        predictions: TimeSeriesDataset = backtest.run(
            predictors=predictors,
            ground_truth=ground_truth,
            start=target.benchmark_start,
            end=target.benchmark_end,
        )
        return scoring_maker(predictions, ground_truth.select_version())

    def optimize(self, experiment_name: str, target_provider: Liander2024TargetProvider) -> HyperParams:
        """Optimize hyperparameters using Optuna over multiple targets.

        In this case we run the optimizer with a single job, parralielization happens at the target level.

        Args:
            experiment_name: Name of the optimization experiment.
            target_provider: The target provider to get targets, predictors and ground truth from.

        Returns:
            The best hyperparameters found during optimization.
        """
        target_backtest_factory = partial(
            self._run_trial_for_target,
            target_provider=target_provider,
            base_config=self.base_config,
            backtest_maker=self._make_backtest,
            scoring_maker=self._score_predictions,
        )

        objective = partial(
            self.objective,
            target_provider=target_provider,
            score_for_target=target_backtest_factory,
            n_jobs=self.n_jobs,
        )

        return self._run_optimization_sequential(
            objective=objective,
            experiment_name=experiment_name,
            n_trials=self.n_trials,
        )

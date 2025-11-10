# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Backtesting pipeline for evaluating energy forecasting models.

Simulates realistic forecasting scenarios by replaying historical data with
proper temporal constraints. Executes prediction and retraining schedules
that mirror operational deployment conditions, ensuring evaluation results
accurately reflect real-world model performance.
"""

from datetime import timedelta
from pathlib import Path

from openstef_beam.benchmarking.benchmarks.liander2024 import Liander2024TargetProvider
from openstef_beam.evaluation.metric_providers import (
    RCRPSSampleWeightedProvider,
)
from openstef_beam.parameter_tuning.greedy_backtester import GreedyBacktestConfig
from openstef_beam.parameter_tuning.models import (
    OptimizationMetric,
    ParameterSpace,
)
from openstef_beam.parameter_tuning.optimizer import GreedyOptimizerConfig, GreedyOptunaOptimizer
from openstef_core.types import LeadTime, Quantile
from openstef_models.presets import (
    ForecastingWorkflowConfig,
    create_forecasting_workflow,
)
from openstef_models.workflows.custom_forecasting_workflow import (
    CustomForecastingWorkflow,
)

horizon = LeadTime.from_string("PT12H")
quantiles = [Quantile(0.1), Quantile(0.3), Quantile(0.5), Quantile(0.7), Quantile(0.9)]
forecaster_name = "lgbmlinear"  # Choose model type: "lgbm", "xgboost", "gblinear", "lgbmlinear", "hybrid", "flatliner"

workflow: CustomForecastingWorkflow = create_forecasting_workflow(
    config=ForecastingWorkflowConfig(
        model_id=f"{forecaster_name}_forecaster_",
        model=forecaster_name,
        horizons=[horizon],
        quantiles=quantiles,
    )
)


model = workflow.model


params = ParameterSpace.get_preset(forecaster_name)


# Set optimization goal
optimization_metric = OptimizationMetric(
    metric=RCRPSSampleWeightedProvider(lower_quantile=0.01, upper_quantile=0.99),
    direction_minimize=True,
)

target_provider = Liander2024TargetProvider(data_dir=Path("../data/liander2024-energy-forecasting-benchmark"))

target = target_provider.get_targets()

predictors = target_provider.get_predictors_for_target(target[0])
ground_truth = target_provider.get_measurements_for_target(target[0])


greedy_backtest_config = GreedyBacktestConfig(
    forecasting_model=model,
    horizon=horizon,
    training_data_length=timedelta(days=90),
    model_train_interval=timedelta(days=31),
    max_lagged_features=timedelta(days=14),
)
optimizer_config = GreedyOptimizerConfig(
    parameter_space=params,
    quantiles=quantiles,
    horizon=horizon,
    forecasting_model=model,
    backtest_config=greedy_backtest_config,
    optimization_metric=optimization_metric,
)
optimizer = GreedyOptunaOptimizer(config=optimizer_config)


best_hyperparams = optimizer.optimize(predictors=predictors, ground_truth=ground_truth)


print("Best hyperparameters found:", best_hyperparams)

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
import sys
from datetime import timedelta
from pathlib import Path

from openstef_beam.backtesting.backtest_forecaster.mixins import BacktestForecasterConfig
from openstef_beam.backtesting.backtest_pipeline import BacktestConfig
from openstef_beam.benchmarking.benchmarks.liander2024 import Liander2024TargetProvider
from openstef_beam.evaluation.metric_providers import RCRPSSampleWeightedProvider
from openstef_beam.parameter_tuning.models import (
    FloatDistribution,
    IntDistribution,
    LGBMLinearParameterSpace,
    OptimizationMetric,
)
from openstef_beam.parameter_tuning.optimizer import (
    RigorousOptimizerConfig,
    RigorousOptunaOptimizer,
)
from openstef_core.types import LeadTime, Quantile
from openstef_models.presets import ForecastingWorkflowConfig, create_forecasting_workflow
from openstef_models.workflows.custom_forecasting_workflow import CustomForecastingWorkflow

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(stream=sys.stdout)
logger.addHandler(handler)


single_target = False

horizon = LeadTime.from_string("PT12H")
quantiles = [Quantile(0.1), Quantile(0.3), Quantile(0.5), Quantile(0.7), Quantile(0.9)]
forecaster_name = "lgbmlinear"  # Choose model type: "lgbm", "xgboost", "gblinear", "lgbmlinear", "hybrid", "flatliner"

# Extract pre-defined forecasting model from preset (pre-processing, forecasting, post-processing)
workflow: CustomForecastingWorkflow = create_forecasting_workflow(
    config=ForecastingWorkflowConfig(
        model_id=f"{forecaster_name}_forecaster_",
        model=forecaster_name,
        horizons=[horizon],
        quantiles=quantiles,
    )
)
model = workflow.model

# Define hyperparameter search space
params = LGBMLinearParameterSpace(
    n_estimators=IntDistribution(low=10, high=100),
    learning_rate=FloatDistribution(low=0.01, high=0.5, log=True),
    num_leaves=IntDistribution(low=5, high=120),
    max_depth=IntDistribution(low=1, high=2),
    max_bin=IntDistribution(low=10, high=70),
    reg_lambda=FloatDistribution(low=1e-4, high=10, log=True),
    colsample_bytree=FloatDistribution(low=0.1, high=1.0),
    min_data_in_leaf=IntDistribution(low=2, high=30),
    min_data_in_bin=IntDistribution(low=2, high=30),
    min_child_weight=FloatDistribution(low=1e-2, high=3, log=True),
)

# Set optimization goal
optimization_metric = OptimizationMetric(
    metric=RCRPSSampleWeightedProvider(lower_quantile=0.01, upper_quantile=0.99),
    direction_minimize=True,
)

# Load target provider with historical data
target_provider = Liander2024TargetProvider(data_dir=Path("../data/liander2024-energy-forecasting-benchmark"))

# Create the backtest configuration
backtest_config = BacktestConfig(
    prediction_sample_interval=timedelta(minutes=15),
    predict_interval=timedelta(hours=6),
    train_interval=timedelta(days=7),
)

# Configure the backtest forecaster
backtest_forecaster_config = BacktestForecasterConfig(
    requires_training=True,
    horizon_length=timedelta(hours=13),
    horizon_min_length=timedelta(hours=11),
    predict_context_length=timedelta(days=14),  # Context needed for lag features
    predict_context_min_coverage=0.5,
    training_context_length=timedelta(days=90),  # Three months of training data
    training_context_min_coverage=0.5,
    predict_sample_interval=timedelta(minutes=15),
)

# Create the optimizer configuration
optimizer_config = RigorousOptimizerConfig(
    parameter_space=params,
    quantiles=quantiles,
    horizon=horizon,
    forecasting_model=model,
    backtest_config=backtest_config,
    backtest_forecaster_config=backtest_forecaster_config,
    optimization_metric=optimization_metric,
    n_trials=100,
    n_jobs=6,
    timeout=7200,  # 2 hours
)

optimizer = RigorousOptunaOptimizer(config=optimizer_config)
if single_target:
    target = target_provider.get_targets()[0]
    predictors = target_provider.get_predictors_for_target(target=target)
    ground_truth = target_provider.get_measurements_for_target(target=target)
    best_hyperparams = optimizer.optimize_dataset(predictors=predictors, ground_truth=ground_truth)

else:
    best_hyperparams = optimizer.optimize_target_provider(target_provider=target_provider)

msg = f"{forecaster_name} - Best hyperparameters found: {best_hyperparams}"
logger.info(msg)

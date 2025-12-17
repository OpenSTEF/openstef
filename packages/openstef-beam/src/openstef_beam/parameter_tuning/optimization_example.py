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
    LGBMParameterSpace,
    OptimizationMetric,
)
from openstef_beam.parameter_tuning.optimizer import (
    OptimizerConfig,
    OptunaOptimizer,
)
from openstef_core.types import LeadTime, Quantile
from openstef_models.presets import ForecastingWorkflowConfig

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(stream=sys.stdout)
logger.addHandler(handler)


single_target = False

horizons = [LeadTime.from_string("PT12H")]
quantiles = [Quantile(0.1), Quantile(0.3), Quantile(0.5), Quantile(0.7), Quantile(0.9)]
forecaster_name = "lgbm"  # Choose model type: "lgbm", "xgboost", "gblinear", "lgbmlinear", "hybrid", "flatliner"


# Base Forecasting Workflow Config
base_config = ForecastingWorkflowConfig(
    model_id="common_model_",
    run_name=None,
    model=forecaster_name,
    horizons=horizons,
    quantiles=quantiles,
    model_reuse_enable=True,
    mlflow_storage=None,
    radiation_column="shortwave_radiation",
    rolling_aggregate_features=["mean", "median", "max", "min"],
    wind_speed_column="wind_speed_80m",
    pressure_column="surface_pressure",
    temperature_column="temperature_2m",
    relative_humidity_column="relative_humidity_2m",
    energy_price_column="EPEX_NL",
)


# Define hyperparameter search space
params = LGBMParameterSpace(
    learning_rate=FloatDistribution(low=0.01, high=0.5, log=True),
    num_leaves=IntDistribution(low=5, high=120),
    max_depth=IntDistribution(low=1, high=3),
    reg_lambda=0.2,
)

# Set optimization goal
optimization_metric = OptimizationMetric(
    metric=RCRPSSampleWeightedProvider(lower_quantile=0.01, upper_quantile=0.99),
    direction_minimize=True,
)

# Load target provider with historical data
target_provider = Liander2024TargetProvider(
    data_dir=Path("../data/liander2024-energy-forecasting-benchmark"),
)

# Create the backtest configuration
backtest_config = BacktestConfig(
    prediction_sample_interval=timedelta(minutes=15),
    predict_interval=timedelta(hours=6),
    train_interval=timedelta(days=7),
)

# Configure the backtest forecaster
backtest_forecaster_config = BacktestForecasterConfig(
    requires_training=True,
    predict_length=timedelta(hours=13),
    predict_min_length=timedelta(hours=11),
    predict_context_length=timedelta(days=14),  # Context needed for lag features
    predict_context_min_coverage=0.5,
    training_context_length=timedelta(days=90),  # Three months of training data
    training_context_min_coverage=0.5,
    predict_sample_interval=timedelta(minutes=15),
)


# Create the optimizer configuration
optimizer_config = OptimizerConfig(
    base_config=base_config,
    parameter_space=params,
    backtest_config=backtest_config,
    backtest_forecaster_config=backtest_forecaster_config,
    optimization_metric=optimization_metric,
    n_jobs=4,
    n_trials=20,
)

optimizer = OptunaOptimizer(config=optimizer_config)
if single_target:
    target = target_provider.get_targets()[0]
    predictors = target_provider.get_predictors_for_target(target=target)
    ground_truth = target_provider.get_measurements_for_target(target=target)
    best_hyperparams = optimizer.optimize_dataset(
        predictors=predictors, ground_truth=ground_truth, experiment_name=target.name
    )

else:
    best_hyperparams = optimizer.optimize_target_provider(
        experiment_name="Liander2024 Benchmark", target_provider=target_provider
    )

msg = f"{forecaster_name} - Best hyperparameters found: {best_hyperparams}"
logger.info(msg)

# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0
import json
from datetime import timedelta
from pathlib import Path

from openstef_beam.backtesting.backtest_forecaster.mixins import (
    BacktestForecasterConfig,
)
from openstef_beam.backtesting.backtest_pipeline import BacktestConfig
from openstef_beam.benchmarking.benchmarks.liander2024 import (
    Liander2024TargetProvider,
)
from openstef_beam.evaluation.metric_providers import (
    RCRPSSampleWeightedProvider,
)
from openstef_beam.parameter_tuning.models import (
    FloatDistribution,
    IntDistribution,
    OptimizationMetric,
    ParameterSpace,
)
from openstef_beam.parameter_tuning.optimizer import (
    OptimizerConfig,
    OptunaOptimizer,
)
from openstef_core.types import LeadTime, Quantile
from openstef_models.presets import (
    ForecastingWorkflowConfig,
    create_forecasting_workflow,
)
from openstef_models.workflows.custom_forecasting_workflow import (
    CustomForecastingWorkflow,
)

model_names = ["lgbm", "xgboost"]

# Per-model parameter spaces (examples; adjust)
param_spaces = {
    "lgbm": ParameterSpace(
        # learning_rate=FloatDistribution(low=1e-3, high=1e-3, log=True),
        num_leaves=IntDistribution(low=5, high=6)
    ),
    "xgboost": ParameterSpace(max_depth=IntDistribution(low=2, high=3)),
}
# Create the backtest configuration
backtest_config = BacktestConfig(
    prediction_sample_interval=timedelta(minutes=15),
    predict_interval=timedelta(hours=6),
    train_interval=timedelta(days=7),
)

# Configure the backtest forecaster
backtest_forecaster_config = BacktestForecasterConfig(
    requires_training=True,
    horizon_length=timedelta(days=7),  # How often the model retrains
    horizon_min_length=timedelta(minutes=15),
    predict_context_length=timedelta(days=14),  # Context needed for lag features
    predict_context_min_coverage=0.5,
    training_context_length=timedelta(days=90),  # Three months of training data
    training_context_min_coverage=0.5,
    predict_sample_interval=timedelta(minutes=15),
)

# Set optimization goal
optimization_metric = OptimizationMetric(
    metric=RCRPSSampleWeightedProvider(lower_quantile=0.01, upper_quantile=0.99),
    direction_minimize=True,
)
results = {}

for name in model_names:
    print(f"Optimizing model: {name}")
    # Build a workflow configured for this model (copy/update base config as needed)
    wf_config = ForecastingWorkflowConfig(
        model_id=f"{name}_test_",
        model=name,
        horizons=[LeadTime.from_string("PT12H")],
        quantiles=[Quantile(0.5)],
    )
    workflow = create_forecasting_workflow(config=wf_config)
    model = workflow.model

    optimizer_config = OptimizerConfig(
        parameter_space=param_spaces[name],
        forecaster=name,
        quantiles=[Quantile(0.1), Quantile(0.5), Quantile(0.9)],
        horizon=LeadTime.from_string("PT12H"),
        forecasting_model=model,
        backtest_config=backtest_config,
        backtest_forecaster_config=backtest_forecaster_config,
        optimization_metric=optimization_metric,
    )
    target_provider = Liander2024TargetProvider(data_dir=Path("../data/liander2024-energy-forecasting-benchmark"))

    target = target_provider.get_targets()[0]
    predictors = target_provider.get_predictors_for_target(target)
    ground_truth = target_provider.get_measurements_for_target(target)
    optimizer = OptunaOptimizer(config=optimizer_config)
    best_params = optimizer.optimize(predictors=predictors, ground_truth=ground_truth)

    results[name] = best_params

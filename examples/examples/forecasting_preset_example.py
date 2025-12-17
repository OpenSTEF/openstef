"""Configuring Model Pipeline Example.

====================================

This example demonstrates how to configure and use a complete forecasting pipeline
in OpenSTEF. It shows how to:

1. Create synthetic time series data for demonstration
2. Configure a full forecasting model with preprocessing and postprocessing
3. Set up model storage for persistence
4. Use the workflow pattern for training and prediction

The example uses a ConstantMedianForecaster with feature engineering including
holiday features, lag transforms, and data scaling. This represents a typical
OpenSTEF forecasting setup that can be adapted for real-world use cases.

Key Components:
    - VersionedTimeSeriesDataset: Time series data structure
    - ForecastingModel: Complete forecasting pipeline
    - FeaturePipeline: Preprocessing with holidays and lags
    - LocalModelStorage: File-based model persistence
    - CustomForecastingWorkflow: High-level orchestration

This example is useful for understanding how to integrate all OpenSTEF components
into a working forecasting system.
"""

# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

import logging
from datetime import timedelta
from pathlib import Path

from openstef_beam.analysis.plots import ForecastTimeSeriesPlotter
from openstef_core.datasets import ForecastDataset
from openstef_core.testing import create_synthetic_forecasting_dataset
from openstef_core.types import LeadTime, Q
from openstef_models.integrations.mlflow import MLFlowStorage
from openstef_models.presets import ForecastingWorkflowConfig, create_forecasting_workflow

logging.basicConfig(level=logging.INFO, format="[%(asctime)s][%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

workspace_dir = Path(__file__).parent.resolve()

# Create synthetic time series data
dataset = create_synthetic_forecasting_dataset(
    length=timedelta(days=90),
    wind_influence=-10.0,
    temp_influence=5.0,
    radiation_influence=-7.0,
    stochastic_influence=2.0,
    sample_interval=timedelta(hours=1),
)

workflow = create_forecasting_workflow(
    config=ForecastingWorkflowConfig(
        model_id="gblinear_forecaster_v1",
        model="gblinear",
        horizons=[LeadTime.from_string("PT36H")],
        quantiles=[Q(0.5), Q(0.1), Q(0.9)],
        mlflow_storage=MLFlowStorage(
            tracking_uri=str(workspace_dir / "mlflow_tracking"),
            local_artifacts_path=workspace_dir / "mlflow_tracking_artifacts",
        ),
    )
)

logger.info("Starting model training")
result = workflow.fit(dataset)
if result is not None:
    logger.info("Full eval result:\n%s", result.metrics_full.to_dataframe())

    if result.metrics_test is not None:
        logger.info("Test result:\n%s", result.metrics_test.to_dataframe())

logger.info("Starting forecasting")
forecast: ForecastDataset = workflow.predict(dataset)

print(forecast.data.tail())

# Plot the result
logger.info("Storing forecast plot to forecast_plot.html")
fig = (
    ForecastTimeSeriesPlotter()
    .add_measurements(measurements=dataset.select_version().data["load"])
    .add_model(model_name="gblinear", forecast=forecast.median_series, quantiles=forecast.quantiles_data)
    .plot()
)

fig.write_html("forecast_plot.html")  # pyright: ignore[reportUnknownMemberType]

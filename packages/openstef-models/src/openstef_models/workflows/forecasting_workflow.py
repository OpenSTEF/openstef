# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""High-level workflow orchestration for forecasting operations.

Provides a complete forecasting workflow that combines model management,
callback execution, and optional model persistence. Acts as the main
entry point for production forecasting systems.
"""

from datetime import datetime
from typing import Self

from pydantic import Field

from openstef_core.base_model import BaseModel
from openstef_core.datasets import TimeSeriesDataset, VersionedTimeSeriesDataset
from openstef_core.datasets.validated_datasets import ForecastDataset
from openstef_core.exceptions import NotFittedError
from openstef_models.mixins import ModelIdentifier, ModelStorage, PredictorCallback
from openstef_models.mixins.callbacks import WorkflowContext
from openstef_models.models.forecasting_model import ForecastingModel


class ForecastingCallback(
    PredictorCallback["ForecastingWorkflow", VersionedTimeSeriesDataset | TimeSeriesDataset, ForecastDataset]
):
    """Base callback interface for monitoring forecasting workflow lifecycle events.

    Provides hooks at key stages of the forecasting process to enable custom
    functionality such as logging, metrics collection, model validation,
    data preprocessing, and integration with monitoring systems.

    All methods have default no-op implementations, so subclasses only need
    to override the specific events they care about.

    Example:
        Creating a logging callback:

        >>> class LoggingCallback(ForecastingCallback):
        ...     def on_fit_start(self, pipeline, dataset):
        ...         print(f"Starting training with {len(dataset.data)} samples")
        ...
        ...     def on_predict_end(self, pipeline, dataset, forecasts):
        ...         print(f"Generated {len(forecasts.data)} forecasts")
        >>>
        >>> callback = LoggingCallback()
        >>> workflow = ForecastingWorkflow(model, callbacks=callback) # doctest: +SKIP
    """


class ForecastingWorkflow(BaseModel):
    """Complete forecasting workflow with model management and lifecycle hooks.

    Orchestrates the full forecasting process by combining a ForecastingModel
    with callback execution and optional model persistence. Provides the main
    interface for production forecasting systems where models need to be
    trained, saved, loaded, and used for prediction with monitoring.

    Invariants:
        - Callbacks are executed at appropriate lifecycle stages
        - Model fitting and prediction delegate to the underlying ForecastingModel
        - Storage operations (if configured) maintain model persistence

    Example:
        Basic workflow with callbacks:

        >>> import pandas as pd
        >>> import numpy as np
        >>> from datetime import timedelta
        >>> from openstef_core.datasets import VersionedTimeSeriesDataset
        >>> from openstef_core.types import LeadTime, Q
        >>> from openstef_models.models.forecasting.constant_median_forecaster import (
        ...     ConstantMedianForecaster, ConstantMedianForecasterConfig
        ... )
        >>> from openstef_models.models.forecasting_model import ForecastingModel
        >>> from openstef_models.transforms import FeatureEngineeringPipeline
        >>>
        >>> # Create sample data
        >>> dataset = VersionedTimeSeriesDataset.from_dataframe(
        ...     data=pd.DataFrame({
        ...         "load": np.random.default_rng(42).standard_normal(size=48),
        ...         "timestamp": pd.date_range("2025-01-01", periods=48, freq="h"),
        ...         "available_at": pd.date_range("2025-01-01", periods=48, freq="h"),
        ...     }),
        ...     sample_interval=timedelta(hours=1),
        ... )
        >>>
        >>> # Create model and workflow
        >>> horizons = [LeadTime.from_string("PT24H")]
        >>> model = ForecastingModel(
        ...     preprocessing=FeatureEngineeringPipeline(horizons=horizons),
        ...     forecaster=ConstantMedianForecaster(
        ...         config=ConstantMedianForecasterConfig(
        ...             horizons=horizons, quantiles=[Q(0.5)]
        ...         )
        ...     )
        ... )
        >>>
        >>> class LoggingCallback(ForecastingCallback):
        ...     def on_fit_end(self, context, data):
        ...         print("Model training completed")
        >>>
        >>> workflow = ForecastingWorkflow(model=model, model_id="my_model", callbacks=LoggingCallback())
        >>> workflow.fit(dataset)
        Model training completed
        >>> forecasts = workflow.predict(dataset)
        >>> len(forecasts.data) > 0
        True

        Loading from storage with fallback:

        >>> workflow = ForecastingWorkflow.from_storage(
        ...     model_id="production_model_v1",
        ...     storage=my_storage,
        ...     default_model_factory=lambda: create_default_model()
        ... ) # doctest: +SKIP
    """

    model: ForecastingModel = Field(description="The forecasting model to use.")
    callbacks: ForecastingCallback = Field(default_factory=ForecastingCallback)
    model_id: ModelIdentifier = Field(...)

    def fit(
        self,
        data: VersionedTimeSeriesDataset | TimeSeriesDataset,
        data_val: VersionedTimeSeriesDataset | TimeSeriesDataset | None = None,
    ):
        """Train the forecasting model with callback execution.

        Executes the complete training workflow including pre-fit callbacks,
        model training, and post-fit callbacks.

        Args:
            data: Training dataset for the forecasting model.
            data_val: Optional validation dataset for model tuning.
        """
        context: WorkflowContext[ForecastingWorkflow] = WorkflowContext(workflow=self)

        self.callbacks.on_fit_start(context=context, data=data)

        self.model.fit(data=data, data_val=data_val)

        self.callbacks.on_fit_end(context=context, data=data)

    def predict(
        self, data: VersionedTimeSeriesDataset | TimeSeriesDataset, forecast_start: datetime | None = None
    ) -> ForecastDataset:
        """Generate forecasts with callback execution.

        Executes the complete prediction workflow including pre-prediction callbacks,
        model prediction, and post-prediction callbacks.

        Args:
            data: Input dataset for generating forecasts.
            forecast_start: Optional start time for forecasts.

        Returns:
            Generated forecast dataset.

        Raises:
            NotFittedError: If the underlying model hasn't been trained.
        """
        if not self.model.is_fitted:
            raise NotFittedError(type(self.model).__name__)

        context: WorkflowContext[ForecastingWorkflow] = WorkflowContext(workflow=self)

        self.callbacks.on_predict_start(context=context, data=data)

        forecasts = self.model.predict(data=data, forecast_start=forecast_start)

        self.callbacks.on_predict_end(context=context, data=data, forecasts=forecasts)

        return forecasts

    @classmethod
    def from_storage(
        cls,
        model_id: ModelIdentifier,
        model: ForecastingModel,
        storage: ModelStorage,
        callbacks: ForecastingCallback | None = None,
    ) -> Self:
        """Create a workflow by loading a model from storage with optional fallback.

        Attempts to load a model from storage. If the model is not found and a
        default factory is provided, creates a new model using the factory.

        Args:
            model_id: Identifier for the model to load from storage.
            model: The forecasting model to use for training and prediction.
            storage: Model storage system to load from.
            callbacks: Optional callback handler for the workflow.

        Returns:
            New workflow instance with the loaded or created model.
        """
        model = storage.load_model_state(model_id=model_id, model=model)
        return cls(model=model, callbacks=callbacks or ForecastingCallback(), model_id=model_id)


__all__ = ["ForecastingWorkflow"]

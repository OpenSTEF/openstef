# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""High-level workflow orchestration for forecasting operations.

Provides a complete forecasting workflow that combines model management,
callback execution, and optional model persistence. Acts as the main
entry point for production forecasting systems.
"""

from collections.abc import Callable
from datetime import datetime
from typing import Self

from openstef_core.datasets import TimeSeriesDataset, VersionedTimeSeriesDataset
from openstef_core.datasets.validated_datasets import ForecastDataset
from openstef_core.exceptions import ModelNotFoundError, NotFittedError
from openstef_models.integrations.callbacks import ForecastingCallback
from openstef_models.integrations.model_storage import ModelIdentifier, ModelStorage
from openstef_models.models.forecasting_model import ForecastingModel


class ForecastingWorkflow:
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
        >>> from openstef_models.integrations.callbacks import ForecastingCallback
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
        >>> model = ForecastingModel(
        ...     forecaster=ConstantMedianForecaster(
        ...         config=ConstantMedianForecasterConfig(
        ...             horizons=[LeadTime.from_string("PT24H")],
        ...             quantiles=[Q(0.5)]
        ...         )
        ...     )
        ... )
        >>>
        >>> class LoggingCallback(ForecastingCallback):
        ...     def on_after_fit(self, pipeline, dataset):
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

    model: ForecastingModel
    callbacks: ForecastingCallback
    storage: ModelStorage | None
    model_id: ModelIdentifier

    def __init__(
        self,
        model: ForecastingModel,
        model_id: ModelIdentifier,
        callbacks: ForecastingCallback | None = None,
        storage: ModelStorage | None = None,
    ) -> None:
        """Initialize the forecasting workflow.

        Args:
            model: The forecasting model to use for training and prediction.
            model_id: Optional identifier for model saving operations.
            callbacks: Optional callback handler for lifecycle events.
                If None, creates a default no-op callback.
            storage: Optional model storage for persistence operations.
        """
        self.model = model
        self.callbacks = callbacks or ForecastingCallback()
        self.storage = storage
        self.model_id = model_id

    def fit(self, dataset: VersionedTimeSeriesDataset | TimeSeriesDataset):
        """Train the forecasting model with callback execution.

        Executes the complete training workflow including pre-fit callbacks,
        model training, and post-fit callbacks.

        Args:
            dataset: Training dataset for the forecasting model.
        """
        self.callbacks.on_before_fit(pipeline=self, dataset=dataset)

        self.model.fit(dataset=dataset)

        self.callbacks.on_after_fit(pipeline=self, dataset=dataset)

    def predict(
        self, dataset: VersionedTimeSeriesDataset | TimeSeriesDataset, forecast_start: datetime | None = None
    ) -> ForecastDataset:
        """Generate forecasts with callback execution.

        Executes the complete prediction workflow including pre-prediction callbacks,
        model prediction, and post-prediction callbacks.

        Args:
            dataset: Input dataset for generating forecasts.
            forecast_start: Optional start time for forecasts.

        Returns:
            Generated forecast dataset.

        Raises:
            NotFittedError: If the underlying model hasn't been trained.
        """
        if not self.model.is_fitted:
            raise NotFittedError(type(self.model).__name__)

        self.callbacks.on_before_predict(pipeline=self, dataset=dataset)

        forecasts = self.model.predict(dataset=dataset, forecast_start=forecast_start)

        self.callbacks.on_after_predict(pipeline=self, dataset=dataset, forecasts=forecasts)

        if self.storage is not None:
            self.storage.save_model(model_id=self.model_id, model=self.model)

        return forecasts

    @classmethod
    def from_storage(
        cls,
        model_id: ModelIdentifier,
        storage: ModelStorage,
        callbacks: ForecastingCallback | None = None,
        default_model_factory: Callable[[], ForecastingModel] | None = None,
    ) -> Self:
        """Create a workflow by loading a model from storage with optional fallback.

        Attempts to load a model from storage. If the model is not found and a
        default factory is provided, creates a new model using the factory.

        Args:
            model_id: Identifier for the model to load from storage.
            storage: Model storage system to load from.
            callbacks: Optional callback handler for the workflow.
            default_model_factory: Optional factory function to create a default
                model if the specified model is not found in storage.

        Returns:
            New workflow instance with the loaded or created model.

        Raises:
            ModelNotFoundError: If model is not found and no default factory provided.
        """
        try:
            model = storage.load_model(model_id=model_id)
        except ModelNotFoundError:
            if default_model_factory is None:
                raise

            model = default_model_factory()

        return cls(model=model, callbacks=callbacks, storage=storage, model_id=model_id)


__all__ = ["ForecastingWorkflow"]

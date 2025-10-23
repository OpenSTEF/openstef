# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Standard interfaces for integrating forecasting models with backtesting.

Establishes the contract between any forecasting model and the backtesting
framework, ensuring consistent behavior across different model types while
supporting both single and batch prediction modes.
"""

from abc import abstractmethod
from collections.abc import Sequence
from datetime import datetime, timedelta

from pydantic import Field

from openstef_beam.backtesting.restricted_horizon_timeseries import RestrictedHorizonVersionedTimeSeries
from openstef_core.base_model import BaseConfig
from openstef_core.datasets import TimeSeriesDataset
from openstef_core.types import Quantile


class BacktestForecasterConfig(BaseConfig):
    """Configuration parameters for backtesting forecasting models.

    Defines the operational constraints and requirements for a forecasting model
    during backtesting simulations. Controls data availability requirements,
    prediction horizons, and training schedules.
    """

    requires_training: bool = Field(description="Whether the model needs to be trained.")

    predict_sample_interval: timedelta = Field(
        default=timedelta(minutes=15), description="Time interval between prediction samples."
    )

    horizon_length: timedelta = Field(description="Length of the prediction horizon.")
    horizon_min_length: timedelta = Field(description="Minimum horizon length that can be predicted.")

    predict_context_length: timedelta = Field(description="Length of the prediction context.")
    predict_context_min_coverage: float = Field(
        description="Minimum number of values that can be NaN in the prediction context."
    )
    training_context_length: timedelta = Field(description="Length of the training context.")
    training_context_min_coverage: float = Field(
        description="Minimum number of values that can be NaN in the training context."
    )


class BacktestForecasterMixin:
    """Mixin defining the interface for forecasting models in backtesting frameworks.

    Provides the essential contract for models that can be used in backtesting pipelines.
    Implementations must handle training on historical data and generating predictions
    with proper timestamp and availability metadata.

    Key responsibilities:
    - Train models on historical time series data with proper context windows
    - Generate probabilistic forecasts across specified quantiles
    - Handle missing data gracefully and return None when predictions aren't possible
    - Provide consistent output formatting for downstream evaluation

    Implementation requirements:
    - Must implement quantiles property to specify which quantiles are predicted
    - Must implement predict() method for core forecasting logic
    - Should implement fit() method for model training (optional for some models)
    - Output predictions must include quantile columns formatted as [quantile_PXX]

    Example:
        Basic implementation for a simple forecasting model:

        >>> from openstef_beam.backtesting.backtest_forecaster import BacktestForecasterMixin
        >>> from openstef_core.types import Quantile
        >>> import pandas as pd
        >>>
        >>> class SimpleForecaster(BacktestForecasterMixin):
        ...     def __init__(self, config):
        ...         self.config = config
        ...         self._quantiles = [Quantile(0.1), Quantile(0.5), Quantile(0.9)]
        ...
        ...     @property
        ...     def quantiles(self):
        ...         return self._quantiles
        ...
        ...     def fit(self, data):
        ...         # Train model on historical data
        ...         self.model_params = self._extract_patterns(data)
        ...
        ...     def predict(self, data):
        ...         # Generate predictions for the forecast horizon
        ...         if not self._has_sufficient_data(data):
        ...             return None
        ...
        ...         predictions = self._generate_forecasts(data)
        ...         return self._format_output(predictions)

    Integration with benchmarking:
        Forecaster implementations are typically created by factory functions
        that customize the model for specific targets:

        >>> def create_forecaster(context, target):
        ...     config = BacktestForecasterConfig(
        ...         predict_context_length=timedelta(days=7),
        ...         training_context_length=timedelta(days=365),
        ...         # ... other config parameters
        ...     )
        ...     return SimpleForecaster(config)
        >>>
        >>> # Use in benchmark pipeline
        >>> # benchmark = BenchmarkPipeline(...)
        >>> # benchmark.run(forecaster_factory=create_forecaster)

    Guarantees:
    - Returns None when prediction cannot be performed reliably
    - Provides properly formatted prediction output with timestamp metadata
    - Handles edge cases and missing data gracefully
    - Maintains consistent error handling patterns across implementations
    """

    config: BacktestForecasterConfig

    @property
    @abstractmethod
    def quantiles(self) -> list[Quantile]:
        """Return the list of quantiles that this forecaster predicts."""
        raise NotImplementedError

    def fit(self, data: RestrictedHorizonVersionedTimeSeries) -> None:
        """Handles the training of the model.

        Args:
           data: Time series data with context for training.
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, data: RestrictedHorizonVersionedTimeSeries) -> TimeSeriesDataset | None:
        """Core prediction logic to be implemented by subclasses.

        Args:
            data: Time series data with context for prediction.

        Returns:
            DataFrame with predictions or None if prediction cannot be performed.
                - The predictions should be formatted in quantile columns [quantile_PXX]
                - The index should be the timestamp of the prediction
        """
        raise NotImplementedError

    def predict_versioned(self, data: RestrictedHorizonVersionedTimeSeries) -> TimeSeriesDataset | None:
        """Predicts a versioned time series with the model.

        Guarantees:
            - Returns None if _handle_predict returns None
            - Adds 'timestamp' column from index and 'available_at' from horizon
            - Contains quantile columns formatted as [quantile_PXX]

        Args:
            data: Time series data with context for prediction.

        Returns:
            VersionedTimeSeriesDataset with predictions.
        """
        prediction = self.predict(data)

        return (
            _version_timeseries_by_timestamp(prediction=prediction, available_at=data.horizon)
            if prediction is not None
            else None
        )


class BacktestBatchForecasterMixin:
    """Extension mixin for forecasters that support batch prediction operations.

    Enables efficient processing of multiple prediction requests simultaneously,
    which can significantly improve performance for models that benefit from
    batch operations (e.g., neural networks, GPU-accelerated models).

    Attributes:
        batch_size: Maximum number of predictions to process in a single batch.
    """

    batch_size: int | None = Field(..., description="Batch size for prediction.")

    def predict_batch(self, batch: list[RestrictedHorizonVersionedTimeSeries]) -> Sequence[TimeSeriesDataset | None]:
        """Handles batch prediction across multiple HorizonTransform instances.

        Args:
            batch: List of HorizonTransform instances for batch prediction.

        Returns:
            List of DataFrames with predictions or None for each instance.
        """
        raise NotImplementedError

    def predict_batch_versioned(
        self, batch: list[RestrictedHorizonVersionedTimeSeries]
    ) -> Sequence[TimeSeriesDataset | None]:
        """Predicts a batch of versioned time series with the model.

        Args:
            batch: List of HorizonTransform instances for batch prediction.

        Returns:
            List of VersionedTimeSeriesDataset with predictions.
        """
        predictions = self.predict_batch(batch)

        return [
            _version_timeseries_by_timestamp(prediction=prediction, available_at=data.horizon)
            if prediction is not None
            else None
            for prediction, data in zip(predictions, batch, strict=True)
        ]


def _version_timeseries_by_timestamp(prediction: TimeSeriesDataset, available_at: datetime) -> TimeSeriesDataset:
    """Adds the 'available_at' column to the prediction DataFrame.

    Args:
        prediction: DataFrame with predictions.
        available_at: Timestamp indicating when the prediction is available.

    Returns:
        DataFrame with 'available_at' column added.
    """
    return TimeSeriesDataset(
        data=prediction.data.assign(available_at=available_at),
        sample_interval=prediction.sample_interval,
    )

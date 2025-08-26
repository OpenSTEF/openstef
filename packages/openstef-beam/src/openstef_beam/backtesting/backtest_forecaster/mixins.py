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

from openstef_core.base_model import BaseConfig
from openstef_core.datasets import TimeSeriesDataset, VersionedTimeSeriesDataset
from openstef_core.datasets.versioned_timeseries.accessors import RestrictedHorizonVersionedTimeSeries
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
    """Mixin defining the interface for forecasting models for backtesting.

    Defines the contract for model training and prediction operations.
    Concrete implementations must provide training and prediction logic.

    Guarantees:
       - Properly formatted prediction output with timestamp and available_at fields
       - Consistent error handling patterns
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

    def predict_versioned(self, data: RestrictedHorizonVersionedTimeSeries) -> VersionedTimeSeriesDataset | None:
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
            _version_timeseries_by_horizon(prediction=prediction, horizon=data.horizon)
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
    ) -> Sequence[VersionedTimeSeriesDataset | None]:
        """Predicts a batch of versioned time series with the model.

        Args:
            batch: List of HorizonTransform instances for batch prediction.

        Returns:
            List of VersionedTimeSeriesDataset with predictions.
        """
        predictions = self.predict_batch(batch)

        return [
            _version_timeseries_by_horizon(prediction=prediction, horizon=data.horizon)
            if prediction is not None
            else None
            for prediction, data in zip(predictions, batch, strict=True)
        ]


def _version_timeseries_by_horizon(prediction: TimeSeriesDataset, horizon: datetime) -> VersionedTimeSeriesDataset:
    """Adds the 'available_at' column to the prediction DataFrame.

    Args:
        prediction: DataFrame with predictions.
        horizon: Timestamp indicating when the prediction is available.

    Returns:
        DataFrame with 'available_at' column added.
    """
    prediction_data = prediction.data.reset_index(names=["timestamp"])
    prediction_data["available_at"] = horizon
    return VersionedTimeSeriesDataset(
        data=prediction_data,
        sample_interval=prediction.sample_interval,
    )

# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Multi-horizon forecasting adapter and utilities.

Provides an abstract adapter that converts single-horizon forecasting models into
multi-horizon forecasters. The adapter handles training separate models for each
prediction horizon and combining their outputs.
"""

from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Self, cast, override

import pandas as pd

from openstef_core.base_model import BaseModel
from openstef_core.datasets.validated_datasets import ForecastDataset, ForecastInputDataset
from openstef_core.datasets.validation import validate_same_sample_intervals
from openstef_core.types import LeadTime
from openstef_core.utils.pandas import unsafe_sorted_range_slice_idxs
from openstef_models.models.mixins.forecaster_mixin import (
    BaseForecaster,
    BaseHorizonForecaster,
    ForecasterConfig,
    HorizonForecasterConfig,
)
from openstef_models.models.mixins.stateful_model_mixin import ModelState


class MultiHorizonForecasterConfig[FC: ForecasterConfig](ForecasterConfig):
    """Configuration for multi-horizon forecaster adapters."""

    forecaster_config: FC


class MultiHorizonForecasterState[FC: HorizonForecasterConfig](BaseModel):
    """Serializable state for multi-horizon forecaster adapters."""

    config: MultiHorizonForecasterConfig[FC]
    forecasters: dict[LeadTime, ModelState]


class MultiHorizonForecasterAdapter[
    FC: HorizonForecasterConfig,
    F: BaseHorizonForecaster,
](BaseForecaster, ABC):
    """Abstract adapter converting single-horizon forecasters to multi-horizon.

    This adapter allows any single-horizon forecaster to work across multiple
    prediction horizons. It maintains separate forecaster instances for each
    horizon and coordinates training and prediction across all horizons.

    The adapter handles:
    - Creating individual forecasters for each horizon
    - Training each forecaster with horizon-specific data
    - Combining predictions from all horizons into unified output
    - State serialization and restoration for model persistence

    Subclasses must implement the abstract methods to specify which forecaster
    type to use and how to create forecaster instances.

    Example:
        Creating a multi-horizon adapter:

        >>> from openstef_models.models.mixins import HorizonForecasterConfig, BaseHorizonForecaster
        >>>
        >>> class MyConfig(HorizonForecasterConfig):
        ...     pass
        >>>
        >>> class MySingleHorizonForecaster(BaseHorizonForecaster):
        ...     def __init__(self, config): pass
        ...     def fit_horizon(self, data): pass
        ...     def predict_horizon(self, data): pass
        ...     def get_state(self): return {}
        ...     @classmethod
        ...     def from_state(cls, state): return cls(None)
        ...     @property
        ...     def config(self): return None
        ...     @property
        ...     def is_fitted(self): return True
        >>>
        >>> class MyMultiHorizonForecaster(
        ...     MultiHorizonForecasterAdapter[MyConfig, MySingleHorizonForecaster]
        ... ):
        ...     @classmethod
        ...     def get_forecaster_type(cls):
        ...         return MySingleHorizonForecaster
        ...     @classmethod
        ...     def create_forecaster(cls, config):
        ...         return MySingleHorizonForecaster(config)
    """

    _config: MultiHorizonForecasterConfig[FC]
    _horizon_forecasters: dict[LeadTime, F]

    def __init__(
        self,
        config: MultiHorizonForecasterConfig[FC],
        horizon_forecasters: dict[LeadTime, F],
    ) -> None:
        """Initialize the multi-horizon forecaster adapter.

        Args:
            config: Configuration wrapping the underlying forecaster config.
            horizon_forecasters: Pre-created forecasters for each horizon.
        """
        self._config = config
        self._horizon_forecasters = horizon_forecasters

    @property
    def config(self) -> MultiHorizonForecasterConfig[FC]:
        """Access the multi-horizon forecaster configuration."""
        return self._config

    @property
    def is_fitted(self) -> bool:
        """Check if all horizon forecasters are fitted and ready for predictions."""
        return all(forecaster.is_fitted for forecaster in self._horizon_forecasters.values())

    @classmethod
    @abstractmethod
    def get_forecaster_type(cls) -> type[F]:
        """Return the type of single-horizon forecaster to use.

        Returns:
            The forecaster class that will be instantiated for each horizon.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def create_forecaster(cls, config: FC) -> F:
        """Create a new single-horizon forecaster instance.

        Args:
            config: Configuration for the single-horizon forecaster.

        Returns:
            A new instance of the single-horizon forecaster.
        """
        raise NotImplementedError

    @classmethod
    def create(
        cls,
        config: MultiHorizonForecasterConfig[FC],
    ) -> Self:
        """Create a new multi-horizon forecaster from configuration.

        Creates individual single-horizon forecasters for each specified horizon
        and wraps them in the multi-horizon adapter.

        Args:
            config: Multi-horizon configuration with list of horizons.

        Returns:
            New multi-horizon forecaster ready for training.
        """
        return cls(
            config=config,
            horizon_forecasters={
                lead_time: cls.create_forecaster(config=config.forecaster_config.with_horizon(lead_time))
                for lead_time in config.horizons
            },
        )

    @override
    def get_state(self) -> ModelState:
        return MultiHorizonForecasterState[FC](
            config=self._config,
            forecasters={
                lead_time: forecaster.get_state() for lead_time, forecaster in self._horizon_forecasters.items()
            },
        )

    @classmethod
    @override
    def from_state(cls, state: ModelState) -> Self:
        state = cast(MultiHorizonForecasterState[FC], state)
        forecaster_type = cls.get_forecaster_type()

        return cls(
            config=state.config,
            horizon_forecasters={
                lead_time: forecaster_type.from_state(forecaster_state)
                for lead_time, forecaster_state in state.forecasters.items()
            },
        )

    @override
    def fit(self, input_data: dict[LeadTime, ForecastInputDataset]) -> None:
        for lead_time, forecaster in self._horizon_forecasters.items():
            forecaster.fit_horizon(input_data[lead_time])

    @override
    def predict(self, input_data: dict[LeadTime, ForecastInputDataset]) -> ForecastDataset:
        predictions = {
            lead_time: forecaster.predict_horizon(input_data[lead_time])
            for lead_time, forecaster in self._horizon_forecasters.items()
        }
        return combine_horizon_forecasts(predictions)


def combine_horizon_forecasts(forecasts: dict[LeadTime, ForecastDataset]) -> ForecastDataset:
    """Combine multiple horizon-specific forecasts into a single consolidated forecast.

    Merges forecasts by using each forecast for its specialized time range.
    Shorter horizon forecasts are used for initial periods, progressively
    switching to longer horizon forecasts for later periods.

    Args:
        forecasts: Dictionary mapping lead times to their corresponding forecast
            datasets. All forecasts must have the same sample interval.

    Returns:
        Combined forecast dataset with merged time series data, using the
        earliest forecast start time and common sample interval.

    Raises:
        ValueError: If forecasts dictionary is empty.

    See Also:
        MultiHorizonForecasterAdapter: Main class that uses this function.
        ForecastDataset: The forecast data structure being combined.
    """
    if len(forecasts) == 0:
        raise ValueError("No forecasts to combine")

    sample_interval = validate_same_sample_intervals(forecasts.values())

    timeseries_chunks: list[pd.DataFrame] = []
    last_lead_time = LeadTime(timedelta(0))
    forecast_start: datetime | None = None
    for lead_time in sorted(forecasts.keys()):
        forecast = forecasts[lead_time]
        forecast_start = (
            forecast.forecast_start if forecast_start is None else min(forecast_start, forecast.forecast_start)
        )

        # Slice the data by its lead time chunk.
        # TimeSeries enforce sorting invariants, so we can use binary search.
        forecast_data = forecasts[lead_time].data
        start_chunk_idx, end_chunk_idx = unsafe_sorted_range_slice_idxs(
            data=cast("pd.Series[pd.Timestamp]", forecast_data.index),
            start=forecast.forecast_start + last_lead_time.value,
            end=forecast.forecast_start + lead_time.value,
        )
        chunk = forecast_data.iloc[start_chunk_idx:end_chunk_idx]

        timeseries_chunks.append(chunk)
        last_lead_time = lead_time

    return ForecastDataset(
        data=pd.concat(timeseries_chunks, axis=0), sample_interval=sample_interval, forecast_start=forecast_start
    )

# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Linear component splitter for energy component analysis.

Provides a linear model based component splitter that splits
energy data into predefined components.

The splitter applies a pre-trained model from OpenSTEF V3.4.24 to divide total energy consumption
into three predefined components. Training is currently not supported.
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, override

import joblib
import pandas as pd
from pydantic import Field

from openstef_core.datasets import EnergyComponentDataset, TimeSeriesDataset
from openstef_core.types import EnergyComponentType
from openstef_models.models.component_splitting.component_splitter import ComponentSplitter, ComponentSplitterConfig

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

_logger = logging.getLogger(__name__)


class LinearComponentSplitterModel(Protocol):
    """Protocol for linear component splitter model interface.

    Defines the expected interface for the linear component splitter model loaded from joblib.
    """

    def predict(self, x: "pd.DataFrame | npt.NDArray[np.float64]") -> "npt.NDArray[np.float64]":
        """Predict energy components from input features.

        Args:
            x: Input features as dataframe or numpy array.

        Returns:
            Predicted components as numpy array.
        """
        ...


class LinearComponentSplitterConfig(ComponentSplitterConfig):
    """Configuration for linear component splitter."""

    linear_model_path: Path = Field(
        default=Path(__file__).parent / "linear_component_splitter_model" / "linear_component_splitter_model.z",
        description="Path to the pre-trained linear model file.",
    )
    radiation_column: str = Field(
        default="radiation",
        description="Column name in the input dataset representing radiation.",
    )
    windspeed_100m_column: str = Field(
        default="windspeed_100m",
        description="Column name in the input dataset representing windspeed at 100m.",
    )


class LinearComponentSplitter(ComponentSplitter):
    """Linear component splitter for energy data.

    Provides a linear component splitter that uses a simple linear model to split
    energy data into predefined components. The predefined components are:
    - Wind on shore
    - Solar
    - Other

    The splitter applies a pre-trained model from OpenSTEF V3.4.24 to divide total energy consumption
    into three predefined components. Training is currently not supported.

    Example:
        Basic usage:

        >>> from openstef_core.types import EnergyComponentType
        >>> config = LinearComponentSplitterConfig(
        ...     source_column="total_load",
        ...     components=[EnergyComponentType.SOLAR, EnergyComponentType.WIND, EnergyComponentType.OTHER],
        ... )
        >>> splitter = LinearComponentSplitter(config)
        >>> components = splitter.predict(time_series_data) # doctest: +SKIP
    """

    _config: LinearComponentSplitterConfig
    _model: LinearComponentSplitterModel | None

    def __init__(self, config: LinearComponentSplitterConfig) -> None:
        """Initialize the linear component splitter.

        Args:
            config: Configuration with model path and column names.
        """
        super().__init__()
        self._config = config
        self._model = joblib.load(self.config.linear_model_path)  # type: ignore[reportUnknownMemberType]

    @property
    @override
    def config(self) -> LinearComponentSplitterConfig:
        """Get the splitter configuration.

        Returns:
            Current configuration with component ratios and settings.
        """
        return self._config

    @property
    @override
    def is_fitted(self) -> bool:
        return True

    def _create_input_features(self, data: TimeSeriesDataset) -> pd.DataFrame:
        """Create input features required by the linear model.

        Args:
            data: Input time series dataset with required columns.

        Returns:
            DataFrame with the 3 features needed for linear model prediction:
            radiation, windspeed_100m, and total_load.

        Raises:
            ValueError: If required columns are missing.
        """
        df = data.data

        source_col = self.config.source_column
        radiation_col = self.config.radiation_column
        wind_col = self.config.windspeed_100m_column

        # Create feature dataframe with the expected column names
        input_df = pd.DataFrame(
            {
                "radiation": df[radiation_col],
                "windspeed_100m": df[wind_col],
                "total_load": df[source_col],
            },
            index=df.index,
        )

        # Drop rows with NaN values
        input_df = input_df.dropna()  # pyright: ignore[reportUnknownMemberType]

        if input_df.empty:
            error_msg = "No valid data available for component splitting after dropping NaNs"
            raise ValueError(error_msg)

        return input_df

    @override
    def fit(self, data: TimeSeriesDataset, data_val: TimeSeriesDataset | None = None) -> None:
        """No training supported currently for linear component splitter.

        The linear model is pre-trained and loaded from a file.
        """

    @override
    def predict(self, data: TimeSeriesDataset) -> EnergyComponentDataset:
        """Predict energy components using the linear model.

        Args:
            data: Input time series dataset containing total load, radiation, and windspeed_100m.

        Returns:
            Energy component dataset with wind, solar, and other components.

        Raises:
            ValueError: If required columns are missing or model not loaded.
        """
        if self._model is None:
            raise ValueError("Linear model not loaded")

        input_df = self._create_input_features(data)

        predictions = self._model.predict(input_df)

        # Create component dataframe
        forecasts = pd.DataFrame(
            predictions,
            columns=[EnergyComponentType.WIND, EnergyComponentType.SOLAR],
            index=input_df.index,
        )

        # Clip wind and solar components to be non-negative
        forecasts[EnergyComponentType.SOLAR] = forecasts[EnergyComponentType.SOLAR].clip(lower=0.0)
        forecasts[EnergyComponentType.WIND] = forecasts[EnergyComponentType.WIND].clip(lower=0.0)

        # Calculate "other" component as residual
        forecasts[EnergyComponentType.OTHER] = (
            input_df["total_load"] - forecasts[EnergyComponentType.SOLAR] - forecasts[EnergyComponentType.WIND]
        )

        # Reindex to match original input, fill missing with 0
        components_df = forecasts.reindex(index=data.data.index, fill_value=0.0)

        # Only return requested components
        requested_components = self.config.components
        components_df = components_df[[col for col in requested_components if col in components_df.columns]]

        return EnergyComponentDataset(
            data=components_df,
            sample_interval=data.sample_interval,
        )

# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Constant component splitter for baseline energy component analysis.

Provides a simple baseline component splitter that uses fixed ratios to split
energy data into predefined components. Useful when users already know the
energy distribution of a location or need a simple baseline for comparison
with more sophisticated splitting methods.

The splitter applies constant ratios to divide total energy consumption into
components like solar, wind, or other energy sources. No training is required
since ratios are predefined by the user.
"""

from typing import Self, override

import pandas as pd
from pydantic import Field, field_validator

from openstef_core.datasets import EnergyComponentDataset, TimeSeriesDataset
from openstef_core.mixins import State
from openstef_core.types import EnergyComponentType
from openstef_models.models.component_splitting.component_splitter import ComponentSplitter, ComponentSplitterConfig


class ConstantComponentSplitterConfig(ComponentSplitterConfig):
    """Configuration for constant component splitter.

    Defines fixed ratios for splitting energy data into components. The ratios
    must sum to 1.0 and represent the known distribution of energy sources.
    """

    component_ratios: dict[EnergyComponentType, float] = Field(
        default=...,
        description="Fixed ratios for each energy component, summing to 1.0.",
    )

    @field_validator("component_ratios")
    @classmethod
    def _validate_component_ratios(cls, value: dict[EnergyComponentType, float]) -> dict[EnergyComponentType, float]:
        if sum(value.values()) != 1.0:
            raise ValueError("Component ratios must sum to 1.0.")

        return value


class ConstantComponentSplitter(ComponentSplitter):
    """Constant ratio-based component splitter for energy data.

    Splits energy time series into predefined components using fixed ratios.
    Useful as a baseline splitter when users know the energy distribution or
    need a simple reference implementation.

    The splitter applies constant multiplication factors to the source data
    based on predefined component ratios. No training is required since ratios
    are user-specified. Performance depends entirely on the accuracy of the
    provided ratios.

    Example:
        Basic usage with known solar/wind distribution:

        >>> from openstef_core.types import EnergyComponentType
        >>> config = ConstantComponentSplitterConfig(
        ...     source_column="total_load",
        ...     components=[EnergyComponentType.SOLAR, EnergyComponentType.WIND],
        ...     component_ratios={
        ...         EnergyComponentType.SOLAR: 0.6,
        ...         EnergyComponentType.WIND: 0.4
        ...     }
        ... )
        >>> splitter = ConstantComponentSplitter(config)
        >>> # components = splitter.predict(time_series_data)

        Using predefined configurations:

        >>> solar_splitter = ConstantComponentSplitter.known_solar_park()
        >>> wind_splitter = ConstantComponentSplitter.known_wind_farm()
    """

    _config: ConstantComponentSplitterConfig

    def __init__(self, config: ConstantComponentSplitterConfig) -> None:
        """Initialize the constant component splitter.

        Args:
            config: Configuration with component ratios and source column.
        """
        super().__init__()
        self._config = config

    @property
    @override
    def config(self) -> ConstantComponentSplitterConfig:
        """Get the splitter configuration.

        Returns:
            Current configuration with component ratios and settings.
        """
        return self._config

    @property
    @override
    def is_fitted(self) -> bool:
        return True

    @override
    def fit(self, data: TimeSeriesDataset) -> None:
        pass

    @override
    def predict(self, data: TimeSeriesDataset) -> EnergyComponentDataset:
        source = data.data[self.config.source_column]

        components: dict[EnergyComponentType, pd.Series] = {
            component: source * self.config.component_ratios.get(component, 0.0) for component in self.config.components
        }

        return EnergyComponentDataset(
            data=pd.DataFrame(components, index=data.data.index),
            sample_interval=data.sample_interval,
        )

    @override
    def to_state(self) -> State:
        return self.config.model_dump(mode="json")

    @override
    def from_state(self, state: State) -> Self:
        config = ConstantComponentSplitterConfig.model_validate(state)
        return self.__class__(config=config)

    @classmethod
    def known_solar_park(cls) -> Self:
        """Create a ConstantComponentSplitter with typical ratios for a solar park.

        Returns:
            Configured ConstantComponentSplitter instance.
        """
        config = ConstantComponentSplitterConfig(
            source_column="load",
            component_ratios={
                EnergyComponentType.SOLAR: 1.0,
            },
        )
        return cls(config=config)

    @classmethod
    def known_wind_farm(cls) -> Self:
        """Create a ConstantComponentSplitter with typical ratios for a wind farm.

        Returns:
            Configured ConstantComponentSplitter instance.
        """
        config = ConstantComponentSplitterConfig(
            source_column="load",
            component_ratios={
                EnergyComponentType.WIND: 1.0,
            },
        )
        return cls(config=config)

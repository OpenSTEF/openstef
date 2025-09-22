# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Base classes and configuration for energy component splitting models.

Provides the foundation for implementing component splitters that divide energy
time series data into different energy sources (solar, wind, etc.). The mixin
defines the standard interface that all component splitters must implement.
"""

from abc import abstractmethod

from pydantic import Field

from openstef_core.base_model import BaseConfig
from openstef_core.datasets import EnergyComponentDataset, TimeSeriesDataset
from openstef_core.mixins import Predictor
from openstef_core.types import EnergyComponentType


class ComponentSplitterConfig(BaseConfig):
    """Configuration for component splitting models.

    Defines the basic parameters needed for any component splitter:
    which column to split and what components to split into.
    """

    source_column: str = Field(
        default="load",
        description="Column name in the input dataset representing the total load/target to be split.",
    )
    components: list[EnergyComponentType] = Field(
        default=list(EnergyComponentType),
        description="List of energy components to split the source column into.",
    )


class ComponentSplitter(Predictor[TimeSeriesDataset, EnergyComponentDataset]):
    """Abstract base class for energy component splitting models.

    Defines the standard interface that all component splitters must implement.
    Component splitters take a time series with total energy consumption and
    split it into different energy components (solar, wind, etc.).

    Implementers must provide:
    - Configuration access via the config property
    - Fitting logic (may be no-op for simple splitters)
    - Prediction logic to perform the actual splitting

    Invariants:
        - is_fitted() must return True before predict() can be called
        - predict() must return components that sum to the original source values
    """

    @property
    @abstractmethod
    def config(self) -> ComponentSplitterConfig:
        """Access the model's configuration parameters.

        Returns:
            Configuration object containing fundamental model parameters.
        """
        raise NotImplementedError("Subclasses must implement config")

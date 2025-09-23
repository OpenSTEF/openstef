# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""High-level component splitting model that orchestrates the complete splitting workflow.

Combines preprocessing, component splitting, and postprocessing into a unified interface.
Handles data transformation and validation while providing consistent component
analysis across different splitting algorithms.
"""

from typing import Any, cast, override

from pydantic import Field

from openstef_core.base_model import BaseModel
from openstef_core.datasets import EnergyComponentDataset, TimeSeriesDataset
from openstef_core.datasets.mixins import Self
from openstef_core.datasets.validation import validate_required_columns
from openstef_core.mixins import State, TransformPipeline
from openstef_models.models.component_splitting.component_splitter import ComponentSplitter, ComponentSplitterConfig


class ComponentSplittingModel(BaseModel, ComponentSplitter):
    """Complete component splitting pipeline combining preprocessing, splitting, and postprocessing.

    Orchestrates the full component splitting workflow by managing data preprocessing,
    component splitting algorithms, and result postprocessing. Provides a unified
    interface for splitting energy time series into different components while
    ensuring data consistency and validation throughout the pipeline.

    Invariants:
        - fit() must be called before predict()
        - Component splitter and preprocessing must be compatible
        - Output components must sum to match input source values

    Example:
        Basic component splitting setup:

        >>> from openstef_models.models.component_splitting.constant_component_splitter import (
        ...     ConstantComponentSplitter, ConstantComponentSplitterConfig
        ... )
        >>> from openstef_core.mixins import TransformPipeline
        >>> from openstef_core.types import EnergyComponentType
        >>>
        >>> # Create a component splitter with known energy ratios
        >>> splitter = ConstantComponentSplitter(
        ...     ConstantComponentSplitterConfig(
        ...         source_column="total_load",
        ...         components=[EnergyComponentType.SOLAR, EnergyComponentType.WIND],
        ...         component_ratios={
        ...             EnergyComponentType.SOLAR: 0.6,
        ...             EnergyComponentType.WIND: 0.4
        ...         }
        ...     )
        ... )
        >>> preprocessing = TransformPipeline()
        >>>
        >>> # Create and train model
        >>> model = ComponentSplittingModel(
        ...     component_splitter=splitter,
        ...     preprocessing=preprocessing,
        ...     source_column="total_load"
        ... )
        >>> model.fit(training_data)  # doctest: +SKIP
        >>>
        >>> # Split components
        >>> components = model.predict(new_data) # doctest: +SKIP
    """

    preprocessing: TransformPipeline[TimeSeriesDataset] = Field(
        default_factory=TransformPipeline[TimeSeriesDataset],
        description="Feature engineering pipeline for transforming raw input data into model-ready features.",
    )
    component_splitter: ComponentSplitter = Field(
        default=...,
        description="Underlying component splitting algorithm implementing the splitting logic.",
    )
    postprocessing: TransformPipeline[EnergyComponentDataset] = Field(
        default_factory=TransformPipeline[EnergyComponentDataset],
        description="Postprocessing pipeline for transforming model outputs into final predictions.",
    )
    source_column: str = Field(
        default="load",
        description="Column name in the input dataset representing the total load/target to be split.",
    )

    @property
    @override
    def config(self) -> ComponentSplitterConfig:
        return self.component_splitter.config

    @property
    @override
    def is_fitted(self) -> bool:
        return self.component_splitter.is_fitted

    @override
    def fit(self, data: TimeSeriesDataset, data_val: TimeSeriesDataset | None = None) -> None:
        validate_required_columns(dataset=data, required_columns=[self.source_column, *self.config.components])

        input_data_train = self.preprocessing.fit_transform(data=data)
        input_data_val = self.preprocessing.transform(data=data_val) if data_val else None

        prediction = self.component_splitter.fit_predict(data=input_data_train, data_val=input_data_val)

        self.postprocessing.fit(data=prediction)

    @override
    def predict(self, data: TimeSeriesDataset) -> EnergyComponentDataset:
        validate_required_columns(dataset=data, required_columns=[self.source_column])

        input_data = self.preprocessing.transform(data=data)

        prediction = self.component_splitter.predict(data=input_data)

        return self.postprocessing.transform(data=prediction)

    @override
    def to_state(self) -> State:
        return {
            "source_column": self.source_column,
            "preprocessing": self.preprocessing.to_state(),
            "component_splitter": self.component_splitter.to_state(),
            "postprocessing": self.postprocessing.to_state(),
        }

    @override
    def from_state(self, state: State) -> Self:
        state = cast(dict[str, Any], state)
        return self.__class__(
            source_column=state["source_column"],
            preprocessing=self.preprocessing.from_state(state["preprocessing"]),
            component_splitter=self.component_splitter.from_state(state["component_splitter"]),
            postprocessing=self.postprocessing.from_state(state["postprocessing"]),
        )


__all__ = ["ComponentSplittingModel"]

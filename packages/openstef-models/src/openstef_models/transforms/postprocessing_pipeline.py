# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Postprocessing pipeline for forecast transformations.

Provides specialized pipeline for postprocessing transforms that need access to both
input data and predictions. Unlike standard TransformPipeline, this maintains the input
context throughout the transformation chain.
"""

from collections.abc import Sequence
from typing import Self, cast, override

from pydantic import BaseModel, ConfigDict, Field

from openstef_core.mixins import State, Transform


class PostprocessingTransform[I, O](Transform[tuple[I, O], O]):
    """Base class for postprocessing transforms that operate on (input, output) tuples.

    This transform type receives both the input data and current forecast/output,
    allowing transforms to use input context when modifying predictions. Common use
    cases include adding uncertainty estimates based on input features or applying
    input-dependent corrections.

    Type parameters:
        I: Input data type (e.g., ForecastInputDataset, MultiHorizon)
        O: Output/forecast type (e.g., ForecastDataset)
    """


class PostprocessingPipeline[I, O](BaseModel, PostprocessingTransform[I, O]):
    """Sequential pipeline of postprocessing transformations.

    Applies multiple PostprocessingTransform instances in order, passing both the
    original input data and the evolving forecast through each transform. This allows
    transforms to use input context when modifying predictions.

    Unlike TransformPipeline which threads data through (out1 → in2 → out2 → in3),
    this pipeline maintains the input and threads only the output:
        (input, out1) → transform1 → out1'
        (input, out1') → transform2 → out2'
        ...

    Invariants:
        - Input data remains unchanged throughout the pipeline
        - Transforms are applied in order, each receiving original input and current forecast
        - Pipeline is fitted only when all transforms are fitted

    Example:
        >>> from openstef_core.datasets import ForecastDataset, ForecastInputDataset
        >>> from openstef_models.transforms.postprocessing import (
        ...     ConfidenceIntervalApplicator,
        ...     QuantileSorter
        ... )
        >>> from openstef_core.types import Quantile
        >>>
        >>> # Create pipeline with multiple postprocessing transforms
        >>> pipeline = PostprocessingPipeline[ForecastInputDataset, ForecastDataset](
        ...     transforms=[
        ...         ConfidenceIntervalApplicator(
        ...             quantiles=[Quantile(0.1), Quantile(0.5), Quantile(0.9)]
        ...         ),
        ...         QuantileSorter(),
        ...     ]
        ... )
        >>>
        >>> # Fit on validation data
        >>> pipeline.fit((validation_input, validation_predictions))  # doctest: +SKIP
        >>>
        >>> # Transform new predictions using input context
        >>> final_forecast = pipeline.transform((new_input, raw_predictions)) # doctest: +SKIP
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    transforms: Sequence[PostprocessingTransform[I, O]] = Field(
        default=[],
        description="Sequence of postprocessing transforms to apply. If empty, returns forecast unchanged.",
    )

    @override
    def to_state(self) -> State:
        return [transform.to_state() for transform in self.transforms]

    @override
    def from_state(self, state: State) -> Self:
        state = cast(Sequence[State], state)

        return self.__class__(
            transforms=[transform.from_state(state=s) for transform, s in zip(self.transforms, state, strict=True)]
        )

    @property
    @override
    def is_fitted(self) -> bool:
        return all(transform.is_fitted for transform in self.transforms)

    @override
    def fit(self, data: tuple[I, O]) -> None:
        input_data, forecast = data

        # Fit each transform on the evolving forecast while keeping input constant
        for transform in self.transforms:
            forecast = transform.fit_transform(data=(input_data, forecast))

    @override
    def transform(self, data: tuple[I, O]) -> O:
        input_data, forecast = data

        # Apply each transform, threading the forecast through while keeping input constant
        for transform in self.transforms:
            forecast = transform.transform(data=(input_data, forecast))

        return forecast


__all__ = ["PostprocessingPipeline", "PostprocessingTransform"]

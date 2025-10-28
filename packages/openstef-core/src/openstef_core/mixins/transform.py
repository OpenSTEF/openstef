# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Transform base classes for data processing.

This module provides abstract base classes for implementing data transformations
with state management capabilities. Transforms follow the scikit-learn pattern
with separate fit and transform phases, and support serialization through the
Stateful interface.
"""

from abc import abstractmethod
from collections.abc import Sequence
from typing import override

from pydantic import Field

from openstef_core.base_model import BaseModel
from openstef_core.mixins.stateful import Stateful


class Transform[I, O](Stateful):
    """Abstract base class for data transformations.

    This class provides the basic interface for transforms that can be fitted to data
    of type I and then applied to transform it to type O. It follows the scikit-learn pattern
    with separate fit and transform phases, and includes state management capabilities.

    Type parameters:
        I: The input data type.
        O: The output data type.

    Subclasses must implement the is_fitted property, fit method, transform method,
    and the state management methods from Stateful.

    Example:
        Implementing a simple scaling transform:

        >>> from openstef_core.datasets import TimeSeriesDataset
        >>> class ScaleTransform(Transform[TimeSeriesDataset, TimeSeriesDataset]):
        ...     def __init__(self):
        ...         self.scale_factor = None
        ...
        ...     @property
        ...     def is_fitted(self) -> bool:
        ...         return self.scale_factor is not None
        ...
        ...     def fit(self, data: TimeSeriesDataset) -> None:
        ...         self.scale_factor = data.data.max().max()
        ...
        ...     def transform(self, data: TimeSeriesDataset) -> TimeSeriesDataset:
        ...         scaled_data = data.data / self.scale_factor
        ...         return TimeSeriesDataset(scaled_data, data.sample_interval)
    """

    @property
    @abstractmethod
    def is_fitted(self) -> bool:
        """Check if the transform has been fitted."""

    @abstractmethod
    def fit(self, data: I) -> None:
        """Fit the transform to the input data.

        This method should be called before applying the transform to the data.
        It allows the transform to learn any necessary parameters from the data.

        Args:
            data: The input data to fit the transform on.
        """

    @abstractmethod
    def transform(self, data: I) -> O:
        """Transform the input data.

        This method should apply a transformation to the input data and return a new instance.

        Args:
            data: The input data to be transformed.

        Returns:
            A new instance of the transformed data.

        Raises:
            NotFittedError: If the transform has not been fitted yet.
        """

    def fit_transform(self, data: I) -> O:
        """Fit the transform to the data and then transform it.

        This method combines fitting and transforming into a single step.

        Args:
            data: The input data to fit and transform.

        Returns:
            A new instance of the transformed data.
        """
        if not self.is_fitted:
            self.fit(data=data)
        return self.transform(data)


class TransformPipeline[T](BaseModel, Transform[T, T]):
    """Sequential pipeline of transformations.

    Applies multiple transforms in order, fitting each transform
    on the intermediate outputs of the previous transforms. Ensures proper
    error handling and state management across the pipeline.

    Invariants:
        - Transforms are called in order, receiving the output of the previous transform.
        - Pipeline is considered fitted only when all transforms are fitted

    Example:
        Creating and using a transformation pipeline:

        >>> from openstef_core.datasets import TimeSeriesDataset
        >>> # Create an empty pipeline
        >>> pipeline = TransformPipeline[TimeSeriesDataset](transforms=[])
        >>>
        >>> # The pipeline can be used even when empty
        >>> # processed_data = pipeline.transform(data)
    """

    transforms: Sequence[Transform[T, T]] = Field(
        default=[],
        description="Sequence of transforms to apply in sequence. If empty, the pipeline is a nop.",
    )

    @property
    @override
    def is_fitted(self) -> bool:
        """Check if all transforms in the pipeline are fitted.

        Returns:
            True if all transforms are fitted, False otherwise.
        """
        return all(transform.is_fitted for transform in self.transforms)

    @override
    def fit(self, data: T) -> None:
        """Fit all transforms in the pipeline sequentially.

        Args:
            data: Input data to fit the transforms on.
        """
        for transform in self.transforms:
            data = transform.fit_transform(data=data)

    @override
    def transform(self, data: T) -> T:
        """Transform data using all fitted transforms in sequence.

        Args:
            data: Input data to transform.

        Returns:
            Transformed data after applying all transforms.
        """
        for transform in self.transforms:
            data = transform.transform(data=data)
        return data

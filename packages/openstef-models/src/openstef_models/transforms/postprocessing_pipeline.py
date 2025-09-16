# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Pipeline for applying multiple forecast transformations in sequence.

Chains forecast postprocessing operations together, ensuring they are applied
in the correct order and providing unified fit/transform behavior across
the entire pipeline.
"""

from typing import override

from openstef_core.datasets.transforms import ForecastTransform
from openstef_core.datasets.validated_datasets import ForecastDataset
from openstef_core.exceptions import NotFittedError


class PostprocessingPipeline(ForecastTransform):
    """Sequential pipeline of forecast transformations.

    Applies multiple forecast transforms in order, fitting each transform
    on the intermediate outputs of the previous transforms. Ensures proper
    error handling and state management across the pipeline.

    Invariants:
        - Transforms are called in order, receiving the output of the previous transform.
        - Pipeline is considered fitted only when all transforms are fitted

    Example:
        Creating and using a postprocessing pipeline:

        >>> # Create an empty pipeline (no postprocessors implemented yet)
        >>> pipeline = PostprocessingPipeline()
        >>>
        >>> # The pipeline can be used even when empty
        >>> # processed_forecasts = pipeline.transform(forecasts)
    """

    transforms: list[ForecastTransform]

    def __init__(self, transforms: list[ForecastTransform] | None = None) -> None:
        """Initialize the postprocessing pipeline.

        Args:
            transforms: List of forecast transforms to apply in sequence.
                If None, creates an empty pipeline.
        """
        self.transforms = transforms or []

    @property
    @override
    def is_fitted(self) -> bool:
        """Check if all transforms in the pipeline are fitted.

        Returns:
            True if all transforms in the pipeline are fitted, False otherwise.
            Returns True for empty pipelines.
        """
        return all(transform.is_fitted for transform in self.transforms)

    @override
    def fit(self, data: ForecastDataset) -> None:
        for transform in self.transforms:
            data = transform.fit_transform(data)

    @override
    def transform(self, data: ForecastDataset) -> ForecastDataset:
        if not self.is_fitted:
            raise NotFittedError(self.__class__.__name__)

        for transform in self.transforms:
            data = transform.transform(data)
        return data

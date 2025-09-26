# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Transform for dimensionality reduction in time series data.

This module provides dimensionality reduction functionality. With a choice of various methods
from scikit-learn to reduce the number of features in time series datasets.
"""

from typing import TYPE_CHECKING, Any, Literal, Self, cast, override

import pandas as pd
from pydantic import Field, PrivateAttr
from sklearn.decomposition import PCA, FactorAnalysis, FastICA, KernelPCA

from openstef_core.base_model import BaseConfig
from openstef_core.datasets.timeseries_dataset import TimeSeriesDataset
from openstef_core.exceptions import NotFittedError
from openstef_core.mixins import State
from openstef_core.transforms import TimeSeriesTransform

if TYPE_CHECKING:
    import numpy as np


class DimensionalityReduction(BaseConfig, TimeSeriesTransform):
    """Reduce the dimensionality of a given set of features.

    Available methods include:
        - PCA: linear dimensionality reduction into orthogonal components.
        - Factor analysis: linear dimensionality reduction models observed variables as latent factors + Gaussian noise.
        - FastICA: linear dimensionality reduction that maximizes statistical independence among components.
        - KernelPCA: non-linear dimensionality reduction using rbf kernel.

    Example:
        >>> import pandas as pd
        >>> from datetime import timedelta
        >>> from openstef_core.datasets import TimeSeriesDataset
        >>> from openstef_models.transforms.general import DimensionalityReduction
        >>> # Create sample dataset
        >>> data = pd.DataFrame({
        ...     'load': [100, 120, 110, 130, 125],
        ...     'feature1': [1.0, 2.0, 1.5, 2.5, 2.0],
        ...     'feature2': [1.0, 2.0, 1.5, 2.5, 2.0],
        ...     'feature3': [5.0, 11.0, 8.0, 2.0, 11.0]
        ... }, index=pd.date_range('2025-01-01', periods=5, freq='1h'))
        >>> dataset = TimeSeriesDataset(data, timedelta(hours=1))
        >>> # Initialize and apply transform
        >>> dim_reducer = DimensionalityReduction(
        ...     columns={'feature1', 'feature2', 'feature3'},
        ...     method="pca",
        ...     n_components=2,
        ...     random_state=1234
        ... )
        >>> dim_reducer.fit(dataset)
        >>> transformed_dataset = dim_reducer.transform(dataset)
        >>> transformed_dataset.data.head().round(3)
                             component_1  component_2  load
        2025-01-01 00:00:00       -2.383       -1.166   100
        2025-01-01 01:00:00        3.596        0.335   120
        2025-01-01 02:00:00        0.606       -0.416   110
        2025-01-01 03:00:00       -5.414        0.912   130
        2025-01-01 04:00:00        3.596        0.335   125

    """

    columns: list[str] | None = Field(
        default=None,
        description="List of column names to apply dim reduction to. If None, applies to all columns.",
    )
    method: Literal["pca", "factor_analysis", "fastica", "kernel_pca"] = Field(
        default="pca", description="Dimensionality reduction method to use."
    )
    n_components: int = Field(default=2, description="Desired nr of components after reduction.")
    random_state: int | None = Field(default=42, description="Random state for reproducibility.")

    _dimensionality_reducer: PCA | FactorAnalysis | FastICA | KernelPCA = PrivateAttr()
    _is_fitted: bool = PrivateAttr(default=False)

    # Method-specific parameters
    max_iter: int = Field(default=1000, description="Maximum number of iterations for Factor Analysis and FastICA.")

    @property
    @override
    def is_fitted(self) -> bool:
        return self._is_fitted

    @override
    def fit(self, data: TimeSeriesDataset) -> None:
        columns = self.columns or data.data.columns
        self._dimensionality_reducer.fit(data.data[columns])  # type: ignore[reportUnknownMemberType]
        self._is_fitted = True

    @override
    def transform(self, data: TimeSeriesDataset) -> TimeSeriesDataset:
        columns = self.columns or data.data.columns
        if not self._is_fitted:
            raise NotFittedError(self.__class__.__name__)
        transformed_data: np.ndarray = self._dimensionality_reducer.transform(data.data[columns])  # type: ignore[reportUnknownMemberType]
        transformed_data_pd = pd.DataFrame(
            transformed_data,
            index=data.data.index,
            columns=[f"component_{i + 1}" for i in range(self.n_components)],
        )
        untransformed_columns = [feature_name for feature_name in data.feature_names if feature_name not in columns]

        transformed_data_pd = pd.concat(
            [transformed_data_pd, data.data[untransformed_columns]],
            axis=1,
        )

        return TimeSeriesDataset(data=transformed_data_pd, sample_interval=data.sample_interval)

    @override
    def model_post_init(self, context: Any) -> None:
        if self.method == "pca":
            self._dimensionality_reducer = PCA(n_components=self.n_components, random_state=self.random_state)
        elif self.method == "factor_analysis":
            self._dimensionality_reducer = FactorAnalysis(
                n_components=self.n_components, max_iter=self.max_iter, random_state=self.random_state
            )
        elif self.method == "fastica":
            self._dimensionality_reducer = FastICA(
                n_components=self.n_components, max_iter=self.max_iter, random_state=self.random_state
            )
        elif self.method == "kernel_pca":
            self._dimensionality_reducer = KernelPCA(
                n_components=self.n_components, kernel="rbf", random_state=self.random_state
            )  # rbf for non-linear

    @override
    def to_state(self) -> State:
        return cast(
            State,
            {
                "config": self.model_dump(mode="json"),
                "dimensionality_reducer": self._dimensionality_reducer,
                "is_fitted": self._is_fitted,
            },
        )

    @override
    def from_state(self, state: State) -> Self:
        state = cast(dict[str, Any], state)
        instance = self.model_validate(state["config"])
        instance._dimensionality_reducer = state["dimensionality_reducer"]  # noqa: SLF001
        instance._is_fitted = state["is_fitted"]  # noqa: SLF001
        return instance

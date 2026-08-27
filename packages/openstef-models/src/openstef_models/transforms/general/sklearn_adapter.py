# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Adapter that exposes any scikit-learn transformer as a TimeSeriesTransform."""

import importlib
from typing import Any, override

import pandas as pd
from pydantic import Field, PrivateAttr

from openstef_core.base_model import BaseConfig
from openstef_core.datasets import TimeSeriesDataset
from openstef_core.exceptions import MissingExtraError, NotFittedError
from openstef_core.transforms import TimeSeriesTransform
from openstef_models.utils.feature_selection import FeatureSelection


class SklearnTransformAdapter(BaseConfig, TimeSeriesTransform):
    """Adapt any scikit-learn transformer to the OpenSTEF ``TimeSeriesTransform`` interface.

    The transformer is specified by its import path and constructor parameters rather than
    as an object, so the configuration stays serializable (save/load-able). It is fitted on
    the selected feature columns; its output then replaces those columns while the remaining
    columns pass through unchanged. Output column names come from the transformer's
    ``get_feature_names_out()``, so shape-changing transforms (e.g. PCA, one-hot encoders)
    are handled the same way as shape-preserving ones (e.g. scalers).

    ``features_added()`` is populated after ``fit()`` (the output names of some transformers
    are only known once fitted).

    Example:
        >>> import pandas as pd
        >>> from datetime import timedelta
        >>> from openstef_core.datasets import TimeSeriesDataset
        >>> from openstef_models.transforms.general import SklearnTransformAdapter
        >>>
        >>> data = pd.DataFrame(
        ...     {"load": [100.0, 200.0, 300.0]},
        ...     index=pd.date_range("2025-01-01", periods=3, freq="h"),
        ... )
        >>> dataset = TimeSeriesDataset(data, timedelta(hours=1))
        >>> adapter = SklearnTransformAdapter(transformer_class="sklearn.preprocessing.StandardScaler")
        >>> adapter.fit(dataset)
        >>> transformed = adapter.transform(dataset)
        >>> abs(float(transformed.data["load"].mean().round(6)))
        0.0
        >>> adapter.features_added()
        []
    """

    transformer_class: str = Field(
        description="Import path of the scikit-learn transformer, e.g. 'sklearn.preprocessing.StandardScaler'.",
    )
    transformer_params: dict[str, Any] = Field(
        default_factory=dict,
        description="Keyword arguments passed to the transformer's constructor.",
    )
    selection: FeatureSelection = Field(
        default=FeatureSelection.ALL,
        description="Features the transformer is applied to.",
    )

    _transformer: Any = PrivateAttr()
    _is_fitted: bool = PrivateAttr(default=False)
    _added_features: list[str] = PrivateAttr(default_factory=list)

    @property
    @override
    def is_fitted(self) -> bool:
        return self._is_fitted

    @override
    def model_post_init(self, context: Any) -> None:
        # Restrict to scikit-learn to keep the dynamic import from loading arbitrary modules.
        if not self.transformer_class.startswith("sklearn."):
            msg = f"transformer_class must be a scikit-learn transformer (sklearn.*), got {self.transformer_class!r}."
            raise ValueError(msg)
        module_path, _, class_name = self.transformer_class.rpartition(".")
        try:
            module = importlib.import_module(module_path)
        except ImportError as e:
            raise MissingExtraError("sklearn", package="openstef-models") from e
        try:
            transformer_cls = getattr(module, class_name)
        except AttributeError as e:
            msg = f"{class_name!r} was not found in {module_path!r}."
            raise ValueError(msg) from e
        self._transformer = transformer_cls(**self.transformer_params)

    def _output_names(self, features: list[str]) -> list[str]:
        # Prefer the transformer's own output names (handles shape-changing transforms);
        # fall back to the input names for shape-preserving ones.
        if hasattr(self._transformer, "get_feature_names_out"):
            return list(self._transformer.get_feature_names_out(features))
        return list(features)

    @override
    def fit(self, data: TimeSeriesDataset) -> None:
        features = self.selection.resolve(data.feature_names)
        self._transformer.fit(data.data[features])
        output_names = self._output_names(features)
        self._added_features = [name for name in output_names if name not in data.feature_names]
        self._is_fitted = True

    @override
    def transform(self, data: TimeSeriesDataset) -> TimeSeriesDataset:
        if not self._is_fitted:
            raise NotFittedError(self.__class__.__name__)

        features = self.selection.resolve(data.feature_names)
        output_names = self._output_names(features)
        transformed = pd.DataFrame(
            self._transformer.transform(data.data[features]),
            index=data.data.index,
            columns=output_names,
        )

        # Replace the transformed inputs with the transformer's output, keep the rest.
        passthrough = [column for column in data.data.columns if column not in features]
        result = pd.concat([data.data[passthrough], transformed], axis=1)

        return TimeSeriesDataset(data=result, sample_interval=data.sample_interval)

    @override
    def features_added(self) -> list[str]:
        return self._added_features


__all__ = ["SklearnTransformAdapter"]

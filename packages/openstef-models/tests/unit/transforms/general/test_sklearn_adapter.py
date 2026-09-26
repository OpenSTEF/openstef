# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

from datetime import timedelta

import pandas as pd
import pytest

from openstef_core.datasets import TimeSeriesDataset
from openstef_core.exceptions import NotFittedError
from openstef_models.transforms.general import SklearnTransformAdapter
from openstef_models.utils.feature_selection import Include


def _dataset() -> TimeSeriesDataset:
    data = pd.DataFrame(
        {"load": [100.0, 200.0, 300.0], "temperature": [20.0, 25.0, 30.0]},
        index=pd.date_range("2025-01-01", periods=3, freq="h"),
    )
    return TimeSeriesDataset(data, timedelta(hours=1))


def test_shape_preserving_transformer_scales_in_place():
    """A scaler keeps the same columns and reports no added features."""
    dataset = _dataset()
    adapter = SklearnTransformAdapter(transformer_class="sklearn.preprocessing.StandardScaler")

    adapter.fit(dataset)
    result = adapter.transform(dataset)

    assert set(result.data.columns) == {"load", "temperature"}
    assert result.data["load"].mean() == pytest.approx(0.0, abs=1e-9)
    assert result.data["load"].std(ddof=0) == pytest.approx(1.0)
    assert adapter.features_added() == []


def test_shape_changing_transformer_replaces_features_with_outputs():
    """PCA replaces the input features with its components (the added features)."""
    adapter = SklearnTransformAdapter(
        transformer_class="sklearn.decomposition.PCA",
        transformer_params={"n_components": 2, "random_state": 0},
    )
    dataset = _dataset()

    adapter.fit(dataset)
    result = adapter.transform(dataset)

    added = adapter.features_added()
    assert len(added) == 2
    # the two input features are gone, replaced by the two components
    assert set(result.data.columns) == set(added)
    assert "load" not in result.data.columns


def test_unselected_features_pass_through_unchanged():
    """Only the selected features are transformed; the rest pass through untouched."""
    dataset = _dataset()
    adapter = SklearnTransformAdapter(
        transformer_class="sklearn.preprocessing.StandardScaler",
        selection=Include("load"),
    )

    adapter.fit(dataset)
    result = adapter.transform(dataset)

    assert result.data["temperature"].tolist() == [20.0, 25.0, 30.0]
    assert result.data["load"].mean() == pytest.approx(0.0, abs=1e-9)


def test_transform_before_fit_raises():
    """transform() before fit() raises NotFittedError."""
    adapter = SklearnTransformAdapter(transformer_class="sklearn.preprocessing.StandardScaler")

    with pytest.raises(NotFittedError):
        adapter.transform(_dataset())


def test_config_round_trips_and_rebuilds_transformer():
    """The config is serializable and reconstructs an equivalent transformer."""
    adapter = SklearnTransformAdapter(
        transformer_class="sklearn.decomposition.PCA",
        transformer_params={"n_components": 3},
    )

    restored = SklearnTransformAdapter.model_validate(adapter.model_dump())

    assert restored.transformer_class == "sklearn.decomposition.PCA"
    assert restored.transformer_params == {"n_components": 3}
    assert type(restored._transformer).__name__ == "PCA"
    assert restored._transformer.n_components == 3


def test_non_sklearn_transformer_class_raises():
    """A non-sklearn transformer_class is rejected at construction."""
    with pytest.raises(ValueError, match="scikit-learn"):
        SklearnTransformAdapter(transformer_class="not_a_real_module.Nope")


def test_unknown_sklearn_class_raises():
    """A sklearn.* path pointing at a non-existent class is rejected."""
    with pytest.raises(ValueError, match="not found"):
        SklearnTransformAdapter(transformer_class="sklearn.preprocessing.NotARealTransformer")

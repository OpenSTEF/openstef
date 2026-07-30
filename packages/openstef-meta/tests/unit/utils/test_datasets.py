# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

from collections.abc import Callable
from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from openstef_core.datasets.validated_datasets import EnsembleForecastDataset, ForecastDataset, ForecastInputDataset
from openstef_core.types import Quantile
from openstef_meta.utils.datasets import combine_forecast_input_datasets


@pytest.fixture
def forecast_dataset_factory() -> Callable[[], ForecastDataset]:
    def _make() -> ForecastDataset:
        rng = np.random.default_rng()
        df = pd.DataFrame(
            data={
                "quantile_P10": [90, 180, 270],
                "quantile_P50": [100, 200, 300],
                "quantile_P90": [110, 220, 330],
                "load": [100, 200, 300],
            },
            index=pd.to_datetime(
                [
                    "2023-01-01T10:00:00",
                    "2023-01-01T11:00:00",
                    "2023-01-01T12:00:00",
                ]
            ),
        )
        df += rng.normal(0, 1, df.shape)  # Add slight noise to avoid perfect predictions

        df["available_at"] = pd.to_datetime(
            [
                "2023-01-01T09:50:00",
                "2023-01-01T10:55:00",
                "2023-01-01T12:10:00",
            ]
        )

        return ForecastDataset(
            data=df,
            sample_interval=timedelta(hours=1),
            target_column="load",
        )

    return _make


@pytest.fixture
def base_predictions(
    forecast_dataset_factory: Callable[[], ForecastDataset],
) -> dict[str, ForecastDataset]:
    return {
        "model_1": forecast_dataset_factory(),
        "model_2": forecast_dataset_factory(),
    }


@pytest.fixture
def ensemble_dataset(base_predictions: dict[str, ForecastDataset]) -> EnsembleForecastDataset:
    return EnsembleForecastDataset.from_forecast_datasets(base_predictions)


def test_from_ensemble_output(ensemble_dataset: EnsembleForecastDataset):
    # Assert
    assert isinstance(ensemble_dataset, EnsembleForecastDataset)
    assert ensemble_dataset.data.shape == (3, 7)  # 3 timestamps, 2 learners * 3 quantiles + target
    assert set(ensemble_dataset.forecaster_names) == {"model_1", "model_2"}
    assert set(ensemble_dataset.quantiles) == {Quantile(0.1), Quantile(0.5), Quantile(0.9)}


def test_get_base_predictions_for_quantile(ensemble_dataset: EnsembleForecastDataset):
    # Act
    dataset = ensemble_dataset.get_base_predictions_for_quantile(Quantile(0.5))

    # Assert
    assert isinstance(dataset, ForecastInputDataset)
    assert dataset.data.shape == (3, 3)  # 3 timestamps, 2 learners * 1 quantiles + target


def test_combine_forecast_input_datasets_none_returns_input(ensemble_dataset: EnsembleForecastDataset):
    # Arrange
    base = ensemble_dataset.get_base_predictions_for_quantile(Quantile(0.5))

    # Act / Assert — no additional features leaves the input untouched
    assert combine_forecast_input_datasets(base, None) is base


def test_combine_forecast_input_datasets_invalid_join_raises(ensemble_dataset: EnsembleForecastDataset):
    # Arrange
    base = ensemble_dataset.get_base_predictions_for_quantile(Quantile(0.5))
    additional = ForecastInputDataset(
        data=pd.DataFrame({"load": [1.0, 2.0, 3.0]}, index=base.data.index),
        sample_interval=base.sample_interval,
        target_column="load",
    )

    # Act / Assert
    with pytest.raises(NotImplementedError):
        combine_forecast_input_datasets(base, additional, join="outer")


def test_combine_forecast_input_datasets_left_join_keeps_base_index(ensemble_dataset: EnsembleForecastDataset):
    # Arrange — additional features cover only a subset of the base index
    base = ensemble_dataset.get_base_predictions_for_quantile(Quantile(0.5))
    base_index = base.data.index
    additional = ForecastInputDataset(
        data=pd.DataFrame({"load": [1.0, 2.0], "extra": [10.0, 20.0]}, index=base_index[:-1]),
        sample_interval=base.sample_interval,
        target_column="load",
    )

    # Act
    combined = combine_forecast_input_datasets(base, additional, join="left")

    # Assert — base index is authoritative; uncovered row keeps the feature as NaN
    assert combined.data.index.equals(base_index)
    assert bool(combined.data.loc[base_index[-1], "extra"] != combined.data.loc[base_index[-1], "extra"])

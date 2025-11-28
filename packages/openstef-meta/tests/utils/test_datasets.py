from collections.abc import Callable
from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from openstef_core.datasets.validated_datasets import ForecastDataset, ForecastInputDataset, TimeSeriesDataset
from openstef_core.types import Quantile
from openstef_meta.framework.base_learner import BaseLearnerNames
from openstef_meta.utils.datasets import EnsembleForecastDataset


@pytest.fixture
def simple_dataset() -> TimeSeriesDataset:
    return TimeSeriesDataset(
        data=pd.DataFrame(
            data={
                "available_at": pd.to_datetime([
                    "2023-01-01T09:50:00",  # lead time = 10:00 - 09:50 = +10min
                    "2023-01-01T10:55:00",  # lead time = 11:00 - 10:55 = +5min
                    "2023-01-01T12:10:00",  # lead time = 12:00 - 12:10 = -10min
                    "2023-01-01T13:20:00",  # lead time = 13:00 - 13:20 = -20min
                    "2023-01-01T14:15:00",  # lead time = 14:00 - 14:15 = -15min
                    "2023-01-01T14:30:00",  # lead time = 14:00 - 14:30 = -30min
                ]),
                "value1": [10, 20, 30, 40, 50, 55],  # 55 should override 50 for 14:00
            },
            index=pd.to_datetime([
                "2023-01-01T10:00:00",
                "2023-01-01T11:00:00",
                "2023-01-01T12:00:00",
                "2023-01-01T13:00:00",
                # Duplicate timestamp with different availability
                "2023-01-01T14:00:00",
                "2023-01-01T14:00:00",
            ]),
        ),
        sample_interval=timedelta(hours=1),
    )


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
            index=pd.to_datetime([
                "2023-01-01T10:00:00",
                "2023-01-01T11:00:00",
                "2023-01-01T12:00:00",
            ]),
        )
        df += rng.normal(0, 1, df.shape)  # Add slight noise to avoid perfect predictions

        df["available_at"] = pd.to_datetime([
            "2023-01-01T09:50:00",
            "2023-01-01T10:55:00",
            "2023-01-01T12:10:00",
        ])

        return ForecastDataset(
            data=df,
            sample_interval=timedelta(hours=1),
            target_column="load",
        )

    return _make


@pytest.fixture
def base_learner_output(
    forecast_dataset_factory: Callable[[], ForecastDataset],
) -> dict[BaseLearnerNames, ForecastDataset]:

    return {
        "GBLinearForecaster": forecast_dataset_factory(),
        "LGBMForecaster": forecast_dataset_factory(),
    }


@pytest.fixture
def ensemble_dataset(base_learner_output: dict[BaseLearnerNames, ForecastDataset]) -> EnsembleForecastDataset:
    return EnsembleForecastDataset.from_forecast_datasets(base_learner_output)


def test_from_ensemble_output(ensemble_dataset: EnsembleForecastDataset):

    assert isinstance(ensemble_dataset, EnsembleForecastDataset)
    assert ensemble_dataset.data.shape == (3, 7)  # 3 timestamps, 2 learners * 3 quantiles + target
    assert set(ensemble_dataset.model_names) == {"GBLinearForecaster", "LGBMForecaster"}
    assert set(ensemble_dataset.quantiles) == {Quantile(0.1), Quantile(0.5), Quantile(0.9)}


def test_select_quantile(ensemble_dataset: EnsembleForecastDataset):

    dataset = ensemble_dataset.select_quantile(Quantile(0.5))

    assert isinstance(dataset, ForecastInputDataset)
    assert dataset.data.shape == (3, 3)  # 3 timestamps, 2 learners * 1 quantiles + target


def test_select_quantile_classification(ensemble_dataset: EnsembleForecastDataset):

    dataset = ensemble_dataset.select_quantile_classification(Quantile(0.5))

    assert isinstance(dataset, ForecastInputDataset)
    assert dataset.data.shape == (3, 3)  # 3 timestamps, 2 learners * 1 quantiles + target
    assert all(dataset.target_series.apply(lambda x: x in BaseLearnerNames.__args__))  # type: ignore

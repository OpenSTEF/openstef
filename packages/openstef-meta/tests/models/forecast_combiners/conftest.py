# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

from collections.abc import Callable
from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from openstef_core.datasets.validated_datasets import ForecastDataset
from openstef_meta.utils.datasets import EnsembleForecastDataset


@pytest.fixture
def forecast_dataset_factory() -> Callable[[], ForecastDataset]:
    def _make() -> ForecastDataset:
        rng = np.random.default_rng()
        coef = rng.normal(0, 1, 3)

        df = pd.DataFrame(
            data={
                "quantile_P10": np.array([1, 2, 3]) * coef[0],
                "quantile_P50": np.array([1, 2, 3]) * coef[1],
                "quantile_P90": np.array([1, 2, 3]) * coef[2],
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
def ensemble_dataset(forecast_dataset_factory: Callable[[], ForecastDataset]) -> EnsembleForecastDataset:
    base_learner_output = {
        "GBLinearForecaster": forecast_dataset_factory(),
        "LGBMForecaster": forecast_dataset_factory(),
    }

    return EnsembleForecastDataset.from_forecast_datasets(base_learner_output)

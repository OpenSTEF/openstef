# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Testing utilities for openstef-models.

Provides shared test helpers (stub forecasters, fixture factories, pipeline
builders) that are used across openstef-models and openstef-meta test suites.
"""

from datetime import timedelta

import numpy as np
import pandas as pd
from pydantic import PrivateAttr

from openstef_core.datasets import TimeSeriesDataset
from openstef_core.datasets.validated_datasets import ForecastDataset, ForecastInputDataset
from openstef_core.mixins import TransformPipeline
from openstef_core.mixins.predictor import HyperParams
from openstef_core.types import LeadTime, override
from openstef_models.models.forecasting.forecaster import Forecaster
from openstef_models.transforms.time_domain.lags_adder import LagsAdder

__all__ = ["SimpleForecaster", "create_sample_timeseries_dataset", "create_standard_preprocessing"]


class SimpleForecaster(Forecaster):
    """Simple test forecaster that returns predictable values for testing."""

    _is_fitted: bool = PrivateAttr(default=False)

    @property
    @override
    def hparams(self) -> HyperParams:  # noqa: D102
        return HyperParams()

    @property
    @override
    def is_fitted(self) -> bool:  # noqa: D102
        return self._is_fitted

    @override
    def fit(self, data: ForecastInputDataset, data_val: ForecastInputDataset | None = None) -> None:  # noqa: D102, ARG002
        self._is_fitted = True

    @override
    def predict(self, data: ForecastInputDataset) -> ForecastDataset:  # noqa: D102
        # Return predictable forecast values
        forecast_values = {quantile: 100.0 + float(quantile) * 10 for quantile in self.quantiles}
        return ForecastDataset(
            pd.DataFrame(
                {quantile.format(): [forecast_values[quantile]] * len(data.index) for quantile in self.quantiles},
                index=data.index,
            ),
            data.sample_interval,
            data.forecast_start,
        )


def create_sample_timeseries_dataset() -> TimeSeriesDataset:
    """Create a small synthetic TimeSeriesDataset for use in unit tests.

    Returns:
        TimeSeriesDataset with 25 hourly samples of load, temperature, and radiation.
    """
    n_samples = 25
    rng = np.random.default_rng(seed=42)

    data = pd.DataFrame(
        {
            "load": 100.0 + rng.normal(10.0, 5.0, n_samples),
            "temperature": 20.0 + rng.normal(1.0, 0.5, n_samples),
            "radiation": rng.uniform(0.0, 500.0, n_samples),
        },
        index=pd.date_range("2025-01-01 10:00", periods=n_samples, freq="h"),
    )

    return TimeSeriesDataset(data, timedelta(hours=1))


def create_standard_preprocessing(horizons: list[LeadTime]) -> TransformPipeline[TimeSeriesDataset]:
    """Create the standard LagsAdder preprocessing pipeline used in pickle/roundtrip tests.

    Returns:
        TransformPipeline with a 14-day LagsAdder.
    """
    return TransformPipeline(
        transforms=[
            LagsAdder(
                history_available=timedelta(days=14),
                horizons=horizons,
                max_day_lags=7,
                add_trivial_lags=True,
                add_autocorr_lags=False,
            ),
        ]
    )

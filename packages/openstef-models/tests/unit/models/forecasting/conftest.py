# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from openstef_core.datasets import ForecastInputDataset


@pytest.fixture
def sample_forecast_input_dataset() -> ForecastInputDataset:
    """Create sample input dataset for forecaster training and prediction."""
    rng = np.random.default_rng(42)
    num_samples = 14
    start_date = datetime.fromisoformat("2025-01-01T00:00:00")

    feature_1 = rng.normal(loc=0, scale=1, size=num_samples)
    feature_2 = rng.normal(loc=0, scale=1, size=num_samples)
    feature_3 = rng.uniform(low=-1, high=1, size=num_samples)

    return ForecastInputDataset(
        data=pd.DataFrame(
            {
                "load": (feature_1 + feature_2 + feature_3) / 3,
                "feature1": feature_1,
                "feature2": feature_2,
                "feature3": feature_3,
            },
            index=pd.date_range(start=start_date, periods=num_samples, freq="1d"),
        ),
        sample_interval=timedelta(days=1),
        target_column="load",
        forecast_start=start_date + timedelta(days=num_samples // 2),
    )


@pytest.fixture
def sample_dataset_with_weights(sample_forecast_input_dataset: ForecastInputDataset) -> ForecastInputDataset:
    """Create sample dataset with sample weights by adding weights to the base dataset."""
    rng = np.random.default_rng(42)
    num_samples = len(sample_forecast_input_dataset.data)

    # Create varied sample weights (some high, some low)
    sample_weights = rng.uniform(low=0.1, high=2.0, size=num_samples)

    # Add sample weights to existing data
    data_with_weights = sample_forecast_input_dataset.data.copy()
    data_with_weights["sample_weight"] = sample_weights

    return ForecastInputDataset(
        data=data_with_weights,
        sample_interval=sample_forecast_input_dataset.sample_interval,
        target_column=sample_forecast_input_dataset.target_column,
        forecast_start=sample_forecast_input_dataset.forecast_start,
    )

# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

from datetime import datetime, timedelta

import pandas as pd
import pytest

from openstef_core.datasets.validated_datasets import TimeSeriesDataset
from openstef_core.exceptions import FlatlinerDetectedError
from openstef_core.types import LeadTime, Q
from openstef_models.presets import ForecastingWorkflowConfig
from openstef_models.presets.forecasting_workflow import _checks  # noqa: PLC2701
from openstef_models.transforms.validation import FlatlineChecker


@pytest.fixture
def config() -> ForecastingWorkflowConfig:
    return ForecastingWorkflowConfig(
        model_id="single_model_",
        model="gblinear",
        quantiles=[Q(0.1), Q(0.5), Q(0.9)],
        horizons=[LeadTime.from_string("PT36H")],
    )


def _get_flatline_checker(config: ForecastingWorkflowConfig) -> FlatlineChecker:
    for transform in _checks(config):
        if isinstance(transform, FlatlineChecker):
            return transform
    msg = "FlatlineChecker not found in single-model preset checks."
    raise AssertionError(msg)


def test_single_model_preset_enables_error_on_flatliner(config: ForecastingWorkflowConfig) -> None:
    assert _get_flatline_checker(config).error_on_flatliner is True


def test_single_model_preset_raises_on_flatliner_input(config: ForecastingWorkflowConfig) -> None:
    flatliner = TimeSeriesDataset(
        data=pd.DataFrame(
            data={"load": [0.0] * 96},
            index=pd.date_range(datetime.fromisoformat("2025-01-01T00:00:00"), periods=96, freq="15min"),
        ),
        sample_interval=timedelta(minutes=15),
    )

    checker = _get_flatline_checker(config)

    with pytest.raises(FlatlinerDetectedError):
        checker.transform(flatliner)

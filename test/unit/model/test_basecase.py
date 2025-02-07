# SPDX-FileCopyrightText: 2017-2023 Contributors to the OpenSTEF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
import unittest
from datetime import datetime, timedelta, timezone, UTC
from test.unit.utils.base import BaseTestCase
from test.unit.utils.data import TestData

import pandas as pd

from openstef.model.basecase import BaseCaseModel

NOW = datetime.now(timezone.utc)


class TestBaseCaseForecast(BaseTestCase):
    def test_basecase_raises_value_error_too_early_start(self):
        # Test if ValueError is raised when forecast start is earlier than allowed
        forecast_input = TestData.load("reference_sets/307-test-data.csv")
        # Shift example data to match current time interval as code expects data
        # available relative to the current time.
        utc_now = (
            pd.Series(datetime.now(tz=UTC))
            .min()
            .round("15T")
            .to_pydatetime()
        )
        most_recent_date = forecast_input.index.max().round("15T").to_pydatetime()
        delta = utc_now - most_recent_date + timedelta(3)

        forecast_input.index = forecast_input.index.shift(delta, freq=1)

        with self.assertRaises(ValueError):
            BaseCaseModel().predict(forecast_input)

    def test_basecase_raises_value_error_missing_features(self):
        # Test if ValueError is raised when T-7d or T-14 are not pressent.
        forecast_input = TestData.load("reference_sets/307-test-data.csv")
        # Shift example data to match current time interval as code expects data
        # available relative to the current time.
        utc_now = (
            pd.Series(datetime.now(tz=UTC))
            .min()
            .round("15T")
            .to_pydatetime()
        )
        most_recent_date = forecast_input.index.max().round("15T").to_pydatetime()
        delta = (
            utc_now - most_recent_date + timedelta(35)
        )  # This will make it pass the first input vallidation

        forecast_input.index = forecast_input.index.shift(delta, freq=1)
        with self.assertRaises(ValueError):
            BaseCaseModel().predict(forecast_input)


if __name__ == "__main__":
    unittest.main()

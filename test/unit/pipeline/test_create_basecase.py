# SPDX-FileCopyrightText: 2017-2023 Contributors to the OpenSTEF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
from datetime import datetime, timedelta, UTC
from test.unit.utils.base import BaseTestCase
from test.unit.utils.data import TestData

import numpy as np
import pandas as pd

from openstef.exceptions import NoRealisedLoadError
from openstef.pipeline.create_basecase_forecast import create_basecase_forecast_pipeline


class TestBaseCaseForecast(BaseTestCase):
    def setUp(self) -> None:
        super().setUp()
        # Generic prep for test
        self.PJ = TestData.get_prediction_job(pid=307)
        forecast_input = TestData.load("reference_sets/307-test-data.csv")
        # Set last 7 days to nan, just like operationally
        forecast_input.loc[
            forecast_input.index.max() - timedelta(days=7) :, "load"
        ] = np.nan
        # Shift so the input matches 'now'
        offset_seconds = (
            pd.to_datetime(datetime.now(tz=UTC))
            - (forecast_input.index.max() - timedelta(days=7))
        ).total_seconds()
        forecast_input = forecast_input.shift(
            freq="min", periods=int(int(offset_seconds / 60.0 / 15.0) * 15)
        )

        self.forecast_input = forecast_input

    def test_basecase_pipeline_happy_flow(self):
        """Test happy flow - everything should work just fine"""

        base_case_forecast = create_basecase_forecast_pipeline(
            self.PJ, self.forecast_input
        )

        # Test length of the output
        self.assertEqual(len(base_case_forecast), 673)

        # Test available columns
        self.assertEqual(
            [
                "forecast",
                "tAhead",
                "stdev",
                "quantile_P05",
                "quantile_P10",
                "quantile_P30",
                "quantile_P50",
                "quantile_P70",
                "quantile_P90",
                "quantile_P95",
                "forecast_other",
                "quality",
                "pid",
                "customer",
                "description",
                "type",
                "algtype",
            ],
            base_case_forecast.columns.to_list(),
        )

        # Test forecast quality label
        self.assertEqual(base_case_forecast["quality"][0], "not_renewed")

    def test_create_basecase_forecast_pipeline_constant_load(self):
        """Historic load can be constant, basecase should still be possible"""

        # load of inputdata is NaN for last 7 days (since normally these are in the future)
        # Change load of inputdata so at the end,
        # 14 days are constant and then 7 days are nan
        forecast_input = self.forecast_input.copy()
        forecast_input.loc[
            forecast_input.index.max() - timedelta(days=21) :, "load"
        ] = forecast_input.loc[forecast_input.index.max() - timedelta(days=14), "load"]
        forecast_input.loc[
            forecast_input.index.max() - timedelta(days=7) :, "load"
        ] = np.nan

        base_case_forecast = create_basecase_forecast_pipeline(self.PJ, forecast_input)

        # Check for length
        self.assertEqual(len(base_case_forecast), 673)
        self.assertEqual(len(base_case_forecast.dropna()), 673)

    def test_create_basecase_forecast_pipeline_zero_values(self):
        """The basecase forecast should also be made if load is constantly zero"""

        forecast_input = self.forecast_input.copy()
        forecast_input.load = 0 * forecast_input.load

        base_case_forecast = create_basecase_forecast_pipeline(self.PJ, forecast_input)
        self.assertEqual(len(base_case_forecast.dropna()), 673)

    def test_create_basecase_forecast_pipeline_no_historic_load(self):
        """If (almost) no historic load is available, a clear exception should be raised"""
        forecast_input = self.forecast_input.copy().iloc[[-1], :]
        with self.assertRaises(NoRealisedLoadError):
            create_basecase_forecast_pipeline(self.PJ, input_data=forecast_input)

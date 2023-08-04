# SPDX-FileCopyrightText: 2017-2023 Contributors to the OpenSTEF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
from datetime import datetime, timedelta, timezone
import unittest
from test.unit.utils.base import BaseTestCase
from test.unit.utils.data import TestData

import pandas as pd

from openstef.pipeline.create_component_forecast import (
    create_components_forecast_pipeline,
)


class TestComponentForecast(BaseTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.PJ = TestData.get_prediction_job(pid=307)

    @unittest.skip("Does work offline, but not online.")
    def test_component_forecast_pipeline_happy_flow(self):
        # Test happy flow
        data = TestData.load("reference_sets/307-test-data.csv")
        weather = data[["radiation", "windspeed_100m"]]
        forecast_input = TestData.load("forecastdf_test_add_corrections.csv")
        forecast_input["stdev"] = 0

        # Shift example data to match current time interval as code expects data
        # available relative to the current time.
        utc_now = (
            pd.Series(datetime.utcnow().replace(tzinfo=timezone.utc))
            .min()
            .round("15T")
            .to_pydatetime()
        )
        most_recent_date = forecast_input.index.max().ceil("15T").to_pydatetime()
        delta = utc_now - most_recent_date + timedelta(3)

        forecast_input.index = forecast_input.index.shift(delta, freq=1)
        most_recent_date = weather.index.max().ceil("15T").to_pydatetime()
        delta = utc_now - most_recent_date + timedelta(3)
        weather.index = weather.index.shift(delta, freq=1)

        component_forecast = create_components_forecast_pipeline(
            self.PJ, forecast_input, weather
        )

        self.assertEqual(len(component_forecast), 193)
        self.assertEqual(
            component_forecast.columns.to_list(),
            [
                "forecast_wind_on_shore",
                "forecast_solar",
                "forecast_other",
                "pid",
                "customer",
                "description",
                "type",
                "algtype",
            ],
        )

    def test_component_forecast_pipeline_not_all_weather_data_available(self):
        # Test happy flow
        data = TestData.load("reference_sets/307-test-data.csv")
        weather = data[["radiation"]]
        forecast_input = TestData.load("forecastdf_test_add_corrections.csv")
        forecast_input["stdev"] = 0

        # Shift example data to match current time interval as code expects data
        # available relative to the current time.
        utc_now = (
            pd.Series(datetime.utcnow().replace(tzinfo=timezone.utc))
            .min()
            .round("15T")
            .to_pydatetime()
        )
        most_recent_date = forecast_input.index.max().ceil("15T").to_pydatetime()
        delta = utc_now - most_recent_date + timedelta(3)

        forecast_input.index = forecast_input.index.shift(delta, freq=1)
        most_recent_date = weather.index.max().ceil("15T").to_pydatetime()
        delta = utc_now - most_recent_date + timedelta(3)
        weather.index = weather.index.shift(delta, freq=1)
        component_forecast = create_components_forecast_pipeline(
            self.PJ, forecast_input, weather
        )
        # Check if the output matches expectations
        self.assertEqual(
            component_forecast.columns.to_list(),
            ["pid", "customer", "description", "type", "algtype"],
        )
        self.assertEqual(len(component_forecast), 0)

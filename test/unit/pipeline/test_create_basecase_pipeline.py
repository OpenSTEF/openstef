from test.utils import BaseTestCase, TestData
import pytz

import numpy as np
import pandas as pd

from datetime import datetime, timezone, timedelta

from openstf.pipeline.basecase_pipeline import basecase_pipeline

NOW = datetime.now(timezone.utc)


class TestBaseCaseForecast(BaseTestCase):
    def test_basecase_pipeline(self):
        PJ = TestData.get_prediction_job(pid=307)
        forecast_input = TestData.load("reference_sets/307-test-data.csv")

        # Shift example data to match current time interval as code expects data
        # available relative to the current time.
        utc_now = (
            pd.Series(datetime.utcnow().replace(tzinfo=pytz.utc))
            .min()
            .round("15T")
            .to_pydatetime()
        )
        most_recent_date = forecast_input.index.max().ceil("15T").to_pydatetime()
        delta = utc_now - most_recent_date + timedelta(3)

        forecast_input.index = forecast_input.index.shift(delta, freq=1)
        base_case_forecast = basecase_pipeline(PJ, forecast_input)

        # Test output datetime range
        self.assertEqual(
            (utc_now + timedelta(days=14)).replace(tzinfo=pytz.utc),
            base_case_forecast.index.max().to_pydatetime(),
        )
        self.assertEqual(
            (utc_now + timedelta(days=2)).replace(tzinfo=pytz.utc),
            base_case_forecast.index.min().to_pydatetime(),
        )

        # Test available columns
        self.assertEqual(
            [
                "forecast",
                "forecast_other",
                "T-7d",
                "T-14d",
                "quality",
                "pid",
                "customer",
                "description",
                "type",
                "algtype",
            ],
            base_case_forecast.columns.to_list(),
        )

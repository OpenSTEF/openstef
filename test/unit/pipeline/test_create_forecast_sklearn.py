# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
from openstf.model.serializer import PersistentStorageSerializer
import unittest
import pytz
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from test.utils import BaseTestCase, TestData
from unittest.mock import patch

from openstf.pipeline import utils
from openstf.pipeline import create_forecast

NOW = datetime.now(timezone.utc)
PJ = TestData.get_prediction_job(pid=60)

forecast_input = TestData.load("reference_sets/307-test-data.csv")


class TestCreateForecastPipeline(BaseTestCase):
    @patch("openstf.pipeline.utils.datetime")
    def test_forecast_datetime_range(self, datetime_mock):
        datetime_mock.now.return_value = NOW
        # get current date and time UTC
        datetime_utc = NOW
        # Time range for time interval to be predicted
        forecast_start_expected = datetime_utc - timedelta(
            minutes=PJ["resolution_minutes"]
        )
        forecast_end_expected = datetime_utc + timedelta(minutes=PJ["horizon_minutes"])

        (forecast_start, forecast_end,) = utils.generate_forecast_datetime_range(
            resolution_minutes=PJ["resolution_minutes"],
            horizon_minutes=PJ["horizon_minutes"],
        )
        self.assertEqual(forecast_start, forecast_start_expected)
        self.assertEqual(forecast_end, forecast_end_expected)

    @patch("openstf.validation.validation.is_data_sufficient")
    def test_create_forecast_pipeline_incomplete_inputdata(
        self, is_data_sufficient_mock
    ):
        """Test if a fallback forecast is used when input is incomplete"""
        input_data = forecast_input
        is_data_sufficient_mock.return_value = False
        pj = TestData.get_prediction_job(pid=307)
        pj["model_type_group"] = "default"
        model = PersistentStorageSerializer(
            trained_models_folder=Path("./test/trained_models")
        ).load_model(pid=307)
        if not hasattr(model, "standard_deviation"):  # Renamed the attribute
            model.standard_deviation = model.confidence_interval
        forecast = create_forecast.create_forecast_pipeline_core(
            pj=pj, input_data=input_data, model=model
        )
        assert "substituted" in forecast.quality.values

    def test_create_forecast_pipeline_happy_flow(self):
        """Test the happy flow of the predict pipeline, using a previously trained model"""
        # Test happy flow
        self.pj = TestData.get_prediction_job(pid=307)
        self.pj["model_type_group"] = "default"
        self.forecast_input = TestData.load("reference_sets/307-test-data.csv")

        # Shift example data to match current time interval as code expects data
        # available relative to the current time.
        utc_now = (
            pd.Series(datetime.utcnow().replace(tzinfo=pytz.utc))
            .min()
            .floor("15T")
            .to_pydatetime()
        )
        most_recent_date = self.forecast_input.index.max().ceil("15T").to_pydatetime()
        delta = utc_now - most_recent_date + timedelta(3)

        self.forecast_input.index = self.forecast_input.index.shift(delta, freq=1)

        self.pj["feature_names"] = self.forecast_input.columns[1:]

        model = PersistentStorageSerializer(
            trained_models_folder=Path("./test/trained_models")
        ).load_model(pid=307)

        if not hasattr(model, "standard_deviation"):  # Renamed the attribute
            model.standard_deviation = model.confidence_interval
        forecast = create_forecast.create_forecast_pipeline_core(
            pj=self.pj, input_data=self.forecast_input, model=model
        )
        self.assertEqual(len(forecast), 193)
        self.assertEqual(len(forecast.columns), 15)
        self.assertGreater(forecast.forecast.min(), -5)
        self.assertLess(forecast.forecast.max(), 85)
        self.assertLess(
            forecast.index.max().to_pydatetime().replace(tzinfo=pytz.utc),
            (datetime.utcnow() + timedelta(hours=50)).replace(tzinfo=pytz.utc),
        )
        self.assertGreaterEqual(
            forecast.index.min().to_pydatetime().replace(tzinfo=pytz.utc), utc_now
        )


if __name__ == "__main__":
    unittest.main()

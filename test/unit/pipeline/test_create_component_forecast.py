# SPDX-FileCopyrightText: 2017-2023 Contributors to the OpenSTEF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
from datetime import datetime, timedelta, timezone
from pathlib import Path
from test.unit.utils.base import BaseTestCase
from test.unit.utils.data import TestData

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.absolute()

from openstef.model.regressors.dazls import Dazls
from openstef.pipeline.create_component_forecast import (
    create_components_forecast_pipeline,
)


class TestComponentForecast(BaseTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.PJ = TestData.get_prediction_job(pid=307)

    def test_load_dazls_model(self):
        """Version updates of sklearn could break pickle.load
        Let's test for that explicitly
        Assert that loading the old model generates an exception and the new model does not
        """

        old_model_file = (
            PROJECT_ROOT / "openstef/data/dazls_model_3.2.49/dazls_stored_3.2.49.sav"
        )
        new_model_file = str(
            PROJECT_ROOT / "openstef/data/dazls_model_3.4.7/dazls_stored_3.4.7_"
        )

        self.assertRaises(Exception, joblib.load, old_model_file)
        dazls_model = Dazls()

        dazls_model.domain_model = joblib.load(new_model_file + "domain_model.z")
        dazls_model.domain_model_scaler = joblib.load(
            new_model_file + "domain_model_scaler.z"
        )
        dazls_model.domain_model_input_columns = joblib.load(
            new_model_file + "domain_model_features.z"
        )

        dazls_model.adaptation_model = joblib.load(
            new_model_file + "adaptation_model.z"
        )
        dazls_model.adaptation_model_scaler = joblib.load(
            new_model_file + "adaptation_model_scaler.z"
        )
        dazls_model.adaptation_model_input_columns = joblib.load(
            new_model_file + "adaptation_model_features.z"
        )

        dazls_model.target_columns = joblib.load(new_model_file + "target.z")
        dazls_model.target_scaler = joblib.load(new_model_file + "target_scaler.z")

        assert dazls_model

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

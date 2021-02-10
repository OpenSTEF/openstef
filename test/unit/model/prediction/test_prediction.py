# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

import unittest
from datetime import datetime, timedelta, timezone
from test.utils.base import BaseTestCase
from test.utils.data import TestData

import numpy as np
import pandas as pd

from stf.model.prediction.prediction import AbstractPredictionModel

PJ = TestData.get_prediction_job(pid=307)


class TestAbstractPredictionModel(BaseTestCase):

    def test_make_fallback_forecast(self):
        # Load input data
        forecast = TestData.load("fallback_forecast.csv")
        load = TestData.load("fallback_load.csv")
        forc_section = TestData.load("fallback_index.csv")

        fallback_forecast = AbstractPredictionModel.predict_fallback(
            forecast_index=forc_section.index, load=load
        )

        self.assertDataframeEqual(fallback_forecast, forecast)

    def test_make_basecase_forecast(self):
        historic_load = pd.DataFrame(
            index=pd.date_range(
                start=datetime.now(timezone.utc) - timedelta(days=15),
                end=datetime.now(timezone.utc),
                freq="15T",
            )
        )
        historic_load["load"] = np.linspace(1, len(historic_load), len(historic_load))

        forecast = AbstractPredictionModel.make_basecase_forecast(PJ, historic_load)

        # check if index is two days ahead
        self.assertGreater(
            forecast.index.min(),
            datetime.now(timezone.utc) + timedelta(days=2, minutes=-15),
        )
        # check if index is up to 6 days ahead
        self.assertGreater(
            forecast.index.max(), datetime.now(timezone.utc) + timedelta(days=6)
        )
        # check if colnames are correct
        basecase_forecast_columns = [
            "forecast",
            "stdev",
            "pid",
            "description",
            "type",
            "algtype",
            "quality",
            "customer",
        ]
        self.assertListEqual(
            sorted(list(forecast.columns)), sorted(basecase_forecast_columns)
        )

    # This has been moved from feature_engineering but methods are private now
    # def test_add_corrections(self):
    #     forecast = self.test_data.FORCAST_DF
    #     forecast["index"] = pd.to_datetime(forecast["index"])
    #     forecast = forecast.set_index("index")

    #     corrected_forecast = apply_features.add_stdev(
    #         forecast, self.test_data.STDEVBIAS_DF.drop(['bias'], axis=1)
    #     )
    #     expected_corrected_forecast = self.test_data.CORRECTED_FORECAST
    #     self.assertDataframeEqual(
    #         corrected_forecast, expected_corrected_forecast.drop(['bias'], axis=1))


if __name__ == "__main__":
    unittest.main()

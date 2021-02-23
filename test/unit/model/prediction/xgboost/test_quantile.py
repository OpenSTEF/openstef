# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

import unittest
from test.utils.base import BaseTestCase

import pandas as pd
from datetime import timedelta


from stf.model.prediction.xgboost.quantile import (
    QuantileXGBPredictionModel as PredictionModel,
)
from stf.model.general import ForecastType, MLModelType


class TestQuantileXGBPredictionModel(BaseTestCase):
    def test_add_stdev_to_forecast(self):
        quantile_forecast = pd.DataFrame(
            {
                "forecast": [5, 6, 7],
                "quantile_P01": [
                    3.8368260629795796,
                    4.604191275575496,
                    5.371556488171412,
                ],
                "quantile_P10": [4.3592242172277, 5.2310690606732395, 6.10291390411878],
                "quantile_P25": [
                    4.662755124901959,
                    5.595306149882351,
                    6.527857174862743,
                ],
                "quantile_P50": [5.0, 6.0, 7.0],
                "quantile_P75": [
                    5.337244875098041,
                    6.404693850117649,
                    7.472142825137257,
                ],
                "quantile_P90": [5.6407757827723, 6.7689309393267605, 7.89708609588122],
                "quantile_P99": [
                    6.163173937020421,
                    7.395808724424504,
                    8.62844351182859,
                ],
            }
        )

        forecast_with_stdev = PredictionModel._add_standard_deviation_to_forecast(
            quantile_forecast
        )

        self.assertTrue("stdev" in forecast_with_stdev.columns)

    def test_add_quantiles_to_fallback(self):
        """Test if quantiles are added to fallback forecast"""
        # Construct inputs
        forecast_index = pd.to_datetime(
            [
                "2020-01-10 00:00:00",
                "2020-01-10 00:15:00",
                "2020-01-10 01:00:00",
                "2020-01-10 01:15:00",
            ]
        )
        load_data = pd.DataFrame(
            index=forecast_index - timedelta(days=14), data={"load": [10, 20, 30, 40]}
        )

        m = PredictionModel(
            pj=dict(model=MLModelType.XGB_QUANTILE, id=999),
            forecast_type=ForecastType.DEMAND,
        )

        forecast = m.predict_fallback(forecast_index, load_data)

        self.assertTrue("quantile_P90" in forecast.columns)
        self.assertTrue("substituted" in forecast.quality.values)


if __name__ == "__main__":
    unittest.main()

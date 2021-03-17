# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

import unittest
from test.utils.base import BaseTestCase

import pandas as pd

from openstf.model.prediction.xgboost.xgboost import XGBPredictionModel


class TestXGBPredictionModel(BaseTestCase):
    def test_add_quantiles_to_forecast(self):
        stdev_forecast = pd.DataFrame({"forecast": [5, 6, 7], "stdev": [0.5, 0.6, 0.7]})
        pj = {"quantiles": [0.01, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99]}

        pp_forecast = XGBPredictionModel._add_quantiles_to_forecast(
            stdev_forecast, pj["quantiles"]
        )

        expected_new_columns = [
            f"quantile_P{int(q * 100):02d}" for q in pj["quantiles"]
        ]

        for expected_column in expected_new_columns:
            self.assertTrue(expected_column in pp_forecast.columns)


if __name__ == "__main__":
    unittest.main()

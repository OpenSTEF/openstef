# SPDX-FileCopyrightText: 2017-2022 Contributors to the OpenSTEF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

from unittest import TestCase
from unittest.mock import patch

import numpy as np
import pandas as pd

from openstef.model.confidence_interval_applicator import ConfidenceIntervalApplicator


class MockModel:
    """Mock model is created as input model for confidence_interval_applicator.
    Confidence_interval_applicator requires model to have confidence interval,
     standard deviation and predict method to test all methods."""

    confidence_interval = pd.DataFrame()

    standard_deviation = pd.DataFrame(
        {
            "stdev": [1.1, 2.9, 1.6, 1.9, 3.2],
            "hour": [0, 1, 2, 3, 4],
            "horizon": [5, 5, 5, 5, 5],
        }
    )

    @staticmethod
    def predict(input, quantile):
        stdev_forecast = pd.DataFrame({"forecast": [5, 6, 7], "stdev": [0.5, 0.6, 0.7]})
        return stdev_forecast["stdev"].rename(quantile)

    @property
    def can_predict_quantiles(self):
        return True


class TestConfidenceIntervalApplicator(TestCase):
    def setUp(self) -> None:
        self.quantiles = [0.01, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99]
        self.stdev_forecast = pd.DataFrame(
            {"forecast": [5, 6, 7], "stdev": [0.5, 0.6, 0.7]}
        )

    @patch(
        "openstef.model.confidence_interval_applicator.ConfidenceIntervalApplicator._add_quantiles_to_forecast_quantile_regression"
    )
    @patch(
        "openstef.model.confidence_interval_applicator.ConfidenceIntervalApplicator._add_standard_deviation_to_forecast"
    )
    def test_add_confidence_interval_happy_flow_(
        self, mock_stdev_to_forecast, mock_add_quantiles
    ):
        forecast = pd.DataFrame({"forecast": [5, 6, 7]})
        stdev_forecast = pd.DataFrame({"forecast": [5, 6, 7], "stdev": [0.5, 0.6, 0.7]})
        mock_stdev_to_forecast.return_value = stdev_forecast
        expected_quantile_output = {
            "forecast": {5, 6, 7},
            "stdev": {0.5, 0.6, 0.7},
            "quantile_P01": {0.5, 0.6, 0.7},
            "quantile_P10": {0.5, 0.6, 0.7},
            "quantile_P25": {0.5, 0.6, 0.7},
            "quantile_P50": {0.5, 0.6, 0.7},
            "quantile_P75": {0.5, 0.6, 0.7},
            "quantile_P90": {0.5, 0.6, 0.7},
            "quantile_P99": {0.5, 0.6, 0.7},
        }
        mock_add_quantiles.return_value = expected_quantile_output
        pj = {
            "model": "quantile",
            "quantiles": self.quantiles,
        }
        actual_quantile_output = ConfidenceIntervalApplicator(
            MockModel(), stdev_forecast
        ).add_confidence_interval(forecast, pj)
        self.assertEqual(actual_quantile_output, expected_quantile_output)

    def test_add_standard_deviation_to_forecast(self):
        forecast = pd.DataFrame({"forecast": [5, 6, 7]})
        forecast.index = [
            pd.Timestamp(2012, 5, 1, 1, 30),
            pd.Timestamp(2012, 5, 1, 1, 45),
            pd.Timestamp(2012, 5, 1, 2, 00),
        ]
        actual_stdev_forecast = ConfidenceIntervalApplicator(
            MockModel(), self.stdev_forecast
        )._add_standard_deviation_to_forecast(forecast)
        self.assertTrue("stdev" in actual_stdev_forecast.columns)
        self.assertEqual(actual_stdev_forecast["stdev"][0], 2.9)
        self.assertEqual(actual_stdev_forecast["stdev"][1], 2.9)
        self.assertEqual(actual_stdev_forecast["stdev"][2], 1.6)

    def test_add_quantiles_to_forecast(self):
        pj = {"quantiles": self.quantiles}
        pp_forecast = ConfidenceIntervalApplicator(
            MockModel(), self.stdev_forecast
        )._add_quantiles_to_forecast_quantile_regression(
            self.stdev_forecast, pj["quantiles"]
        )

        expected_new_columns = [
            f"quantile_P{int(q * 100):02d}" for q in pj["quantiles"]
        ]

        for expected_column in expected_new_columns:
            self.assertTrue(expected_column in pp_forecast.columns)

    def test_add_quantiles_to_forecast_length_mismatch(self):
        pj = {"quantiles": self.quantiles}
        pp_forecast = ConfidenceIntervalApplicator(
            MockModel(), self.stdev_forecast.iloc[:-1, :]  # do not use last value
        )._add_quantiles_to_forecast_quantile_regression(
            self.stdev_forecast, pj["quantiles"]
        )

        expected_new_columns = [
            f"quantile_P{int(q * 100):02d}" for q in pj["quantiles"]
        ]

        for expected_column in expected_new_columns:
            self.assertTrue(expected_column in pp_forecast.columns)
            # Assert last quantile value is missing
            self.assertTrue(np.isnan(pp_forecast[expected_column].iloc[-1]))

    def test_add_quantiles_to_forecast_default(self):
        pj = {"quantiles": self.quantiles}

        pp_forecast = ConfidenceIntervalApplicator(
            MockModel(), "TEST"
        )._add_quantiles_to_forecast_default(self.stdev_forecast, pj["quantiles"])

        expected_new_columns = [
            f"quantile_P{int(q * 100):02d}" for q in pj["quantiles"]
        ]

        for expected_column in expected_new_columns:
            self.assertTrue(expected_column in pp_forecast.columns)

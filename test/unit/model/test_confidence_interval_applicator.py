# SPDX-FileCopyrightText: 2017-2023 Contributors to the OpenSTEF project <korte.termijn.prognoses@alliander.com> # noqa E501>
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

    can_predict_quantiles_ = True

    def predict(self, input, quantile):
        if self.can_predict_quantiles and quantile not in self.quantiles:
            # When model is trained on quantiles, it should fail if quantile is not in
            # trained quantiles
            raise ValueError("Quantile not in trained quantiles")

        stdev_forecast = pd.DataFrame({"forecast": [5, 6, 7], "stdev": [0.5, 0.6, 0.7]})
        return stdev_forecast["stdev"].rename(quantile)

    @property
    def can_predict_quantiles(self):
        return self.can_predict_quantiles_

    @property
    def quantiles(self):
        return [0.01, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99]


class MockNonQuantileModel(MockModel):
    @property
    def can_predict_quantiles(self):
        return False


class MockModelMultiHorizonStdev(MockModel):
    standard_deviation = pd.DataFrame(
        {
            "stdev": [0.1, 2.9, 1.6, 1.9, 3.2] + [2.1, 5, 8, 12, 14],
            "hour": [0, 1, 2, 3, 4] * 2,
            "horizon": [5, 5, 5, 5, 5] + [10, 10, 10, 10, 10],
        }
    )


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
        self.assertIn("stdev", actual_stdev_forecast.columns)
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
        # Arange
        pj = {"quantiles": self.quantiles}

        # Act
        pp_forecast = ConfidenceIntervalApplicator(
            MockModel(), self.stdev_forecast.iloc[:-1, :]  # do not use last value
        )._add_quantiles_to_forecast_quantile_regression(
            self.stdev_forecast, pj["quantiles"]
        )

        # Assert
        expected_new_columns = [
            f"quantile_P{int(q * 100):02d}" for q in pj["quantiles"]
        ]

        for expected_column in expected_new_columns:
            # Assert the quantiles are available
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

    def test_add_standard_deviation_to_forecast_in_past(self):
        """Forecasts for negative/zero lead times should result in an exploding stdev"""
        forecast = pd.DataFrame({"forecast": [5, 6, 7], "tAhead": [-1.0, 0.0, 1.0]})
        forecast.index = [
            pd.Timestamp(2012, 5, 1, 1, 30),
            pd.Timestamp(2012, 5, 1, 1, 45),
            pd.Timestamp(2012, 5, 1, 2, 00),
        ]

        actual_stdev_forecast = ConfidenceIntervalApplicator(
            MockModelMultiHorizonStdev(), self.stdev_forecast
        )._add_standard_deviation_to_forecast(forecast)
        self.assertIn("stdev", actual_stdev_forecast.columns)
        self.assertGreaterEqual(
            actual_stdev_forecast["stdev"].min(), 0.1
        )  # => MockModel.standard_deviation.stdev.min()
        self.assertLessEqual(
            actual_stdev_forecast["stdev"].max(), 14
        )  # => MockModel.standard_deviation.stdev.max())

    def test_add_quantiles_to_forecast_untrained_quantiles_with_quantile_model(self):
        """For quantile models, the trained quantiles can used if the quantiles of the pj are incompatible"""
        # Set up
        pj = {"quantiles": [0.12, 0.5, 0.65]}  # numbers are arbitrary
        model = MockModel()
        forecast = pd.DataFrame({"forecast": [5, 6, 7], "tAhead": [-1.0, 0.0, 1.0]})
        forecast.index = [
            pd.Timestamp(2012, 5, 1, 1, 30),
            pd.Timestamp(2012, 5, 1, 1, 45),
            pd.Timestamp(2012, 5, 1, 2, 00),
        ]
        model.can_predict_quantiles_ = True
        # Specify expectation
        expected_quantiles = model.quantiles
        expected_columns = [f"quantile_P{int(q * 100):02d}" for q in expected_quantiles]

        # Act
        pp_forecast = ConfidenceIntervalApplicator(
            model, forecast
        ).add_confidence_interval(forecast, pj)

        # Assert
        for expected_column in expected_columns:
            self.assertTrue(expected_column in pp_forecast.columns)

    def test_add_quantiles_to_forecast_untrained_quantiles_with_nonquantile_model(self):
        """For nonquantile models, the quantiles of the pj should be used, also if the model was not trained on those"""
        # Set up
        pj = {"quantiles": [0.12, 0.5, 0.65]}  # numbers are arbitrary
        model = MockModel()
        forecast = pd.DataFrame({"forecast": [5, 6, 7], "tAhead": [-1.0, 0.0, 1.0]})
        forecast.index = [
            pd.Timestamp(2012, 5, 1, 1, 30),
            pd.Timestamp(2012, 5, 1, 1, 45),
            pd.Timestamp(2012, 5, 1, 2, 00),
        ]
        model.can_predict_quantiles_ = False
        # Specify expectation
        expected_quantiles = pj["quantiles"]
        expected_columns = [f"quantile_P{int(q * 100):02d}" for q in expected_quantiles]

        # Act
        pp_forecast = ConfidenceIntervalApplicator(
            model, forecast
        ).add_confidence_interval(forecast, pj)

        # Assert
        for expected_column in expected_columns:
            self.assertTrue(expected_column in pp_forecast.columns)

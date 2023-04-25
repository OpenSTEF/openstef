# SPDX-FileCopyrightText: 2017-2022 Contributors to the OpenSTEF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

import numpy as np
import pandas as pd

from openstef.validation.validation import calc_completeness_features

from datetime import datetime as dt
from pathlib import Path
from test.unit.utils.base import BaseTestCase
from test.unit.utils.data import TestData
from unittest.mock import patch, MagicMock
from mlflow.exceptions import MlflowException
from openstef.model.serializer import MLflowSerializer


class CalcCompletenessTest(BaseTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.pj = TestData.get_prediction_job(pid=307)
        self.data = TestData.load("reference_sets/307-test-data.csv")
        self.train_input = TestData.load("reference_sets/307-train-data.csv")

    datetime_format = "%Y-%m-%dT%H:%M:%S%z"

    def test_complete_dataframe(self):
        df = pd.DataFrame(index=[0, 1], data={"col1": [1, 1]})
        weights = pd.DataFrame(index=["col1"], data={"gain": 1, "weight": 1})
        completeness = calc_completeness_features(df, weights)

        self.assertEqual(completeness, 1.0)

    def test_APX_missing(self):
        df = pd.DataFrame(index=range(2 * 96), data={"APX": [np.nan] * 2 * 96})
        weights = pd.DataFrame(index=["APX"], data={"gain": 1, "weight": 1})

        completeness = calc_completeness_features(df, weights, time_delayed=True)
        self.assertEqual(completeness, 1 / 2)

    def test_incomplete_dataframe(self):
        df = pd.DataFrame(index=[0, 1, 2], data={"col1": [1, np.nan, 3]})
        weights = pd.DataFrame(index=["col1"], data={"gain": 1, "weight": 1})

        completeness = calc_completeness_features(df, weights)
        self.assertEqual(completeness, 2 / 3)

    def test_weighted_dataframe(self):
        df = pd.DataFrame(index=[0, 1], data={"col1": [1, np.nan], "col2": [3, 4]})
        weights = pd.DataFrame(
            index=["col1", "col2"], data={"gain": [1, 1], "weight": [1, 2]}
        )

        completeness = calc_completeness_features(df, weights)
        self.assertEqual(completeness, (1 * 0.5 + 2 * 1) / 3)

    def test_timedelayed_dataframe(self):
        df = pd.DataFrame(
            index=[0, 1, 2],
            data={"T-15min": [1, 2, np.nan], "T-30min": [2, np.nan, np.nan]},
        )
        weights = pd.DataFrame(
            index=["T-15min", "T-30min"], data={"gain": [1, 1], "weight": [1, 1]}
        )

        completeness = calc_completeness_features(df, weights, time_delayed=True)
        self.assertEqual(completeness, 1 / 2)

    def test_timedelayed_incomplete_dataframe(self):
        df = pd.DataFrame(
            index=[0, 1, 3],
            data={"T-15min": [1, np.nan, np.nan], "T-30min": [2, np.nan, np.nan]},
        )  # first nan is unexpected
        weights = pd.DataFrame(
            index=["T-15min", "T-30min"], data={"gain": [1, 1], "weight": [1, 1]}
        )

        completeness = calc_completeness_features(df, weights, time_delayed=True)
        self.assertAlmostEqual(completeness, 1 / 3, places=3)

    def test_timedelayed_advanced_dataframe(self):
        df = pd.DataFrame(
            index=[0, 1, 3],
            data={
                "T-15min": [1, 2, np.nan],
                "T-30min": [2, np.nan, np.nan],
                "col1": [1, np.nan, 2],
            },
        )
        weights = np.array([1, 1, 2])
        completeness = calc_completeness_features(df, weights, time_delayed=True)
        self.assertEqual(completeness, (2 / 3 * 1 + 1 / 3 * 1 + 2 / 3 * 2) / 4)

    def test_homogenise_timeindex_complete(self):
        df_complete = pd.DataFrame(
            {"aggregated": [10, 20, 30]},
            index=pd.to_datetime(
                ["2019-01-01 10:00:00", "2019-01-01 10:05:00", "2019-01-01 10:10:00"]
            ),
        )
        weights = pd.DataFrame(index=["aggregated"], data={"gain": [1], "weight": [1]})

        completeness_df_complete = calc_completeness_features(df_complete, weights)
        self.assertAlmostEqual(completeness_df_complete, 1)

    def test_homogenise_timeindex_incomplete(self):
        df_incomplete = pd.DataFrame(
            {"aggregated": [10, 20, 30, 40]},
            index=pd.to_datetime(
                [
                    "2019-01-01 10:00:00",
                    "2019-01-01 10:05:00",
                    # Note the missing value
                    "2019-01-01 10:15:00",
                    "2019-01-01 10:20:00",
                ]
            ),
        )
        weights = pd.DataFrame(index=["aggregated"], data={"gain": [1], "weight": [1]})
        completeness_df_incomplete = calc_completeness_features(
            df_incomplete, weights, homogenise=True
        )
        completeness_df_incomplete_nothomogenised = calc_completeness_features(
            df_incomplete, weights, homogenise=False
        )
        self.assertAlmostEqual(completeness_df_incomplete, 0.8)
        self.assertAlmostEqual(completeness_df_incomplete_nothomogenised, 1)

    def test_empty_dataframe(self):
        weights = np.array([1])

        empty_compl_homogenise = calc_completeness_features(
            pd.DataFrame(
                data={"load": [], "aggregated": []}, index=pd.DatetimeIndex([])
            ),
            weights,
        )
        empty_compl_nohomogenise = calc_completeness_features(
            pd.DataFrame(
                data={"load": [], "aggregated": []}, index=pd.DatetimeIndex([])
            ),
            weights,
            homogenise=False,
        )

        self.assertAlmostEqual(empty_compl_homogenise, 0.0)
        self.assertAlmostEqual(empty_compl_nohomogenise, 0.0)

    def test_calc_completeness_no_negatives(self):
        """Test added after bug.
        If time delayed is True, T-7d gave a negative weight,
        falsely resulting in a very low completeness"""
        df = pd.DataFrame(
            index=[0, 1, 3],
            data={
                "T-15min": [1, np.nan, np.nan],
                "T-7d": [2, 3, 4],
                "T-24d": [4, 5, 6],
                "col1": [1, np.nan, 2],
            },
        )
        weights = pd.DataFrame(
            index=["T-15min", "T-7d", "T-24d", "col1"],
            data={"gain": [1, 1, 1, 1], "weight": [1, 1, 1, 1]},
        )

        completeness = calc_completeness_features(df, weights, time_delayed=True)
        self.assertEqual(completeness, (1 / 3 * 1 + 1 + 1 + 2 / 3 * 1) / 4)

    def test_calc_completeness_model_feature_importance_as_weights(self):
        weights = pd.DataFrame(
            index=["temp", "IsSunday", "Month", "snowDepth", "windspeed"],
            data={"weight": 0.1}
        )

        data_with_features = pd.DataFrame(
            data={
                "load": [1, 1, 2, 2, 3, 3, 4, 4],
                "IsSunday": [1, 1, 2, 2, 3, 3, 4, 4],
                "temp": [1, 1, 2, 2, 3, 3, 4, 4],
                "snowDepth": [1, 1, 2, 2, 3, 3, 4, 4],
                "windspeed": [1, 1, 2, 2, 3, 3, 4, 4],
                "Month": [1, 1, 2, 2, 3, 3, 4, 4],
                "horizon": [0.25, 47.0, 0.25, 47.0, 0.25, 47.0, 0.25, 47.0],
            },
            index=pd.to_datetime(
                [
                    "2019-01-01 10:00:00",
                    "2019-01-01 10:00:00",
                    "2019-01-01 10:15:00",
                    "2019-01-01 10:15:00",
                    "2019-01-01 10:30:00",
                    "2019-01-01 10:30:00",
                    "2019-01-01 10:45:00",
                    "2019-01-01 10:45:00",
                ]
            ),
        )
        completeness = calc_completeness_features(
            data_with_features, weights, time_delayed=True, homogenise=False
        )
        self.assertEqual(completeness, 1)

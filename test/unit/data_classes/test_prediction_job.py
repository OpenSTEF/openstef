# SPDX-FileCopyrightText: 2017-2023 Contributors to the OpenSTEF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

import unittest
from test.unit.utils.data import TestData

from openstef.data_classes.prediction_job import PredictionJobDataClass
from openstef.enums import LocationColumnName


class TestPredictionJobs(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.pj, self.modelspecs = TestData.get_prediction_job_and_modelspecs(307)

    def test_prediction_job(self):
        self.assertIsInstance(self.pj, PredictionJobDataClass)

    def test_prediction_job_from_dict(self):
        # Arrange
        pj_dict = {
            "id": 307,
            "model": "xgb",
            "model_type_group": "xgb",
            "horizon_minutes": 2880,
            "resolution_minutes": 15,
            "train_components": 1,
            "name": "Neerijnen",
            LocationColumnName.LAT: 51.8336647,
            LocationColumnName.LON: 5.2137814,
            "sid": "NrynRS_10-G_V12_P",
            "created": "2019-04-05 12:08:23",
            "description": "NrynRS_10-G_V12_P+NrynRS_10-G_V13_P+NrynRS_10-G_V14_P+NrynRS_10-G_V15_P+NrynRS_10-G_V16_P+NrynRS_10-G_V17_P+NrynRS_10-G_V18_P+NrynRS_10-G_V20_P+NrynRS_10-G_V21_P+NrynRS_10-G_V22_P+NrynRS_10-G_V23_P+NrynRS_10-G_V24_P+NrynRS_10-G_V25_P",
            "quantiles": [0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95],
            "hyper_params": {
                "subsample": 0.9650102355823993,
                "min_child_weight": 3,
                "max_depth": 6,
                "gamma": 0.1313691782115394,
                "colsample_bytree": 0.8206844265155975,
                "silent": 1,
                "objective": "reg:squarederror",
                "eta": 0.010025843216782565,
                "training_period_days": 90,
            },
            "feature_names": [
                "clearSky_dlf",
                "clearSky_ulf",
            ],
            "forecast_type": "demand",
        }

        # Assert
        pj = PredictionJobDataClass(**pj_dict)

        # Act
        self.assertIsInstance(pj, PredictionJobDataClass)

    def test_prediction_job_from_dict_with_solar_columns(self):
        # Arrange
        pj_dict = {
            "id": 307,
            "model": "xgb",
            "model_type_group": "xgb",
            "horizon_minutes": 2880,
            "resolution_minutes": 15,
            "train_components": 1,
            "name": "Neerijnen",
            LocationColumnName.LAT: 51.8336647,
            LocationColumnName.LON: 5.2137814,
            "sid": "NrynRS_10-G_V12_P",
            "created": "2019-04-05 12:08:23",
            "description": "NrynRS_10-G_V12_P+NrynRS_10-G_V13_P+NrynRS_10-G_V14_P+NrynRS_10-G_V15_P+NrynRS_10-G_V16_P+NrynRS_10-G_V17_P+NrynRS_10-G_V18_P+NrynRS_10-G_V20_P+NrynRS_10-G_V21_P+NrynRS_10-G_V22_P+NrynRS_10-G_V23_P+NrynRS_10-G_V24_P+NrynRS_10-G_V25_P",
            "quantiles": [0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95],
            "hyper_params": {
                "subsample": 0.9650102355823993,
                "min_child_weight": 3,
                "max_depth": 6,
                "gamma": 0.1313691782115394,
                "colsample_bytree": 0.8206844265155975,
                "silent": 1,
                "objective": "reg:squarederror",
                "eta": 0.010025843216782565,
                "training_period_days": 90,
            },
            "feature_names": [
                "clearSky_dlf",
                "clearSky_ulf",
            ],
            "forecast_type": "demand",
        }

        # Assert
        pj = PredictionJobDataClass(**pj_dict)

        # Act
        self.assertIsInstance(pj, PredictionJobDataClass)
        self.assertEqual(pj["id"], 307)
        self.assertEqual(pj["sid"], "NrynRS_10-G_V12_P")

    def test_prediction_job_from_dict_with_wind_columns(self):
        # Arrange
        pj_dict = {
            "id": 307,
            "turbine_type": "test",
            "n_turbines": 3.0,
            "hub_height": 20.0,
            "model": "xgb",
            "model_type_group": "xgb",
            "horizon_minutes": 2880,
            "resolution_minutes": 15,
            "train_components": 1,
            "name": "Neerijnen",
            LocationColumnName.LAT: 51.8336647,
            LocationColumnName.LON: 5.2137814,
            "sid": "NrynRS_10-G_V12_P",
            "created": "2019-04-05 12:08:23",
            "description": "NrynRS_10-G_V12_P+NrynRS_10-G_V13_P+NrynRS_10-G_V14_P+NrynRS_10-G_V15_P+NrynRS_10-G_V16_P+NrynRS_10-G_V17_P+NrynRS_10-G_V18_P+NrynRS_10-G_V20_P+NrynRS_10-G_V21_P+NrynRS_10-G_V22_P+NrynRS_10-G_V23_P+NrynRS_10-G_V24_P+NrynRS_10-G_V25_P",
            "quantiles": [0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95],
            "hyper_params": {
                "subsample": 0.9650102355823993,
                "min_child_weight": 3,
                "max_depth": 6,
                "gamma": 0.1313691782115394,
                "colsample_bytree": 0.8206844265155975,
                "silent": 1,
                "objective": "reg:squarederror",
                "eta": 0.010025843216782565,
                "training_period_days": 90,
            },
            "feature_names": [
                "clearSky_dlf",
                "clearSky_ulf",
            ],
            "forecast_type": "demand",
        }

        # Assert
        pj = PredictionJobDataClass(**pj_dict)

        # Act
        self.assertIsInstance(pj, PredictionJobDataClass)
        self.assertEqual(pj["id"], 307)
        self.assertEqual(pj["turbine_type"], "test")
        self.assertEqual(pj["n_turbines"], 3.0)
        self.assertEqual(pj["hub_height"], 20.0)

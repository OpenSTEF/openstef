# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
from datetime import datetime, timedelta
from pathlib import Path

import unittest
from unittest.mock import MagicMock, patch
from test.utils import TestData
from test.utils import BaseTestCase

import pandas as pd


from openstf.model.serializer import PersistentStorageSerializer
from openstf.pipeline.train_predict_backtest import (
    train_model_and_forecast_back_test)

# define constants
PJ = TestData.get_prediction_job(pid=307)
XGB_HYPER_PARAMS = {
    "subsample": 0.9,
    "min_child_weight": 4,
    "max_depth": 8,
    "gamma": 0.5,
    "colsample_bytree": 0.85,
    "eta": 0.1,
    "training_period_days": 90,
}



class TestTrainBackTestPipeline(BaseTestCase):

    def setUp(self) -> None:
        super().setUp()
        self.pj = TestData.get_prediction_job(pid=307)
        datetime_start = datetime.utcnow() - timedelta(days=90)
        datetime_end = datetime.utcnow()
        self.data_table = TestData.load("input_data_train.pickle").head(8641)
        self.data = pd.DataFrame(
            index=pd.date_range(datetime_start, datetime_end, freq="15T")
        )

        self.train_input = TestData.load("reference_sets/307-train-data.csv")
        self.model = PersistentStorageSerializer(
            trained_models_folder=Path("./test/trained_models")
        ).load_model(pid=307)

    @patch("openstf.pipeline.train_predict_backtest.model")
    @patch("openstf.pipeline.train_predict_backtest.PersistentStorageSerializer")
    def test_train_model_pipeline_core_happy_flow(self, pss_mock):
        """Test happy flow of the train model pipeline

        NOTE this does not explain WHY this is the case?
        The input data should not contain features (e.g. T-7d),
        but it can/should include predictors (e.g. weather data)

        """



        forecast = train_model_and_forecast_back_test(
            pj=self.pj, input_data=self.train_input, trained_models_folder="TEST",
            save_figures_folder="OTHER_TEST")





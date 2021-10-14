# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

import pandas as pd

from openstf.model.model_creator import ModelCreator
from openstf.model.serializer import (
    PersistentStorageSerializer,
    MODEL_FILENAME,
    FOLDER_DATETIME_FORMAT,
)
from test.utils import BaseTestCase, TestData


class TestAbstractModelSerializer(BaseTestCase):
    def test_determine_model_age_from_path(self):
        expected_model_age = 7

        model_datetime = datetime.utcnow() - timedelta(days=expected_model_age)

        model_path = (
            Path(f"{model_datetime.strftime(FOLDER_DATETIME_FORMAT)}") / MODEL_FILENAME
        )

        model_age = PersistentStorageSerializer(
            trained_models_folder="OTHER_TEST"
        )._determine_model_age_from_path(model_path)

        self.assertEqual(model_age, expected_model_age)

    @patch("mlflow.sklearn.log_model")
    @patch("mlflow.log_figure")
    @patch("mlflow.log_params")
    @patch("mlflow.log_metrics")
    @patch("mlflow.set_tag")
    @patch("mlflow.search_runs")
    def test_save_model(
        self,
        mock_search,
        mock_set_tag,
        mock_log_metrics,
        mock_log_params,
        mock_log_figure,
        mock_log_model,
    ):

        model_type = "xgb"
        model = ModelCreator.create_model(model_type)
        pj = TestData.get_prediction_job(pid=307)
        report_mock = MagicMock()
        report_mock.get_metrics.return_value = {"mae", 0.2}
        with self.assertLogs("PersistentStorageSerializer", level="INFO") as captured:
            PersistentStorageSerializer("OTHER_TEST").save_model(
                model=model, pj=pj, report=report_mock
            )
            self.assertRegex(
                captured.records[0].getMessage(), "Model saved with MLflow"
            )

    @patch("mlflow.sklearn.log_model")
    @patch("mlflow.log_figure")
    @patch("mlflow.log_params")
    @patch("mlflow.log_metrics")
    @patch("mlflow.set_tag")
    @patch("mlflow.search_runs")
    def test_save_model_no_previous(
        self,
        mock_search,
        mock_set_tag,
        mock_log_metrics,
        mock_log_params,
        mock_log_figure,
        mock_log_model,
    ):

        model_type = "xgb"
        model = ModelCreator.create_model(model_type)
        pj = TestData.get_prediction_job(pid=307)
        report_mock = MagicMock()
        report_mock.get_metrics.return_value = {"mae", 0.2}
        mock_search.return_value = pd.DataFrame(columns=["run_id"])
        with self.assertLogs("PersistentStorageSerializer", level="INFO") as captured:
            PersistentStorageSerializer("OTHER_TEST").save_model(
                model=model, pj=pj, report=report_mock
            )
            self.assertRegex(
                captured.records[0].getMessage(), "No previous model found in MLflow"
            )

    def test_determine_model_age_from_MLflow_run(self):
        ts = pd.Timestamp("2021-01-25 00:00:00")
        run = pd.DataFrame(
            {
                "end_time": [
                    ts,
                ],
                "col1": [
                    1,
                ],
            }
        ).iloc[0]
        days = PersistentStorageSerializer("OTHER_TEST")._determine_model_age_from_mlflow_run(run)
        self.assertGreater(days, 7)

    def test_determine_model_age_from_MLflow_run_exception(self):
        ts = pd.Timestamp("2021-01-25 00:00:00")
        # Without .iloc it will not be able to get the age, resulting in a error
        run = pd.DataFrame(
            {
                "end_time": [
                    ts,
                ],
                "col1": [
                    1,
                ],
            }
        )
        with self.assertLogs(
            "PersistentStorageSerializer", level="WARNING"
        ) as captured:
            days = PersistentStorageSerializer("OTHER_TEST")._determine_model_age_from_mlflow_run(
                run
            )
        self.assertRegex(
            captured.records[0].getMessage(),
            "Could not get model age. Returning infinite age!",
        )
        self.assertEqual(days, float("inf"))

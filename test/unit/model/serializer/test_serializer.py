# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock, PropertyMock
import tempfile
from distutils.dir_util import copy_tree

import pandas as pd

from openstf.metrics.reporter import Report
from openstf.data_classes.model_specifications import ModelSpecificationDataClass
from openstf.model.model_creator import ModelCreator
from openstf.model.serializer import (
    MLflowSerializer
)
from test.utils import BaseTestCase, TestData


class TestAbstractModelSerializer(BaseTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.pj, self.modelspecs = TestData.get_prediction_job_and_modelspecs(pid=307)

    @patch("openstf.data_classes.model_specifications.ModelSpecificationDataClass")
    @patch("mlflow.search_runs")
    @patch("mlflow.sklearn.load_model")
    def test_serializer_feature_names_keyerror(
        self, mock_load, mock_search_runs, mock_modelspecs
    ):
        mock_search_runs.return_value = pd.DataFrame(
            data={
                "run_id": [1, 2],
                "artifact_uri": ["path1", "path2"],
                "end_time": [
                    datetime.utcnow() - timedelta(days=2),
                    datetime.utcnow() - timedelta(days=3),
                ],
            }
        )
        mock_modelspecs.return_value = self.modelspecs
        type(mock_load.return_value).feature_names = PropertyMock(return_value=None)
        loaded_model, modelspecs = MLflowSerializer(
            trained_models_folder="./test/trained_models"
        ).load_model(307)
        self.assertIsInstance(modelspecs, ModelSpecificationDataClass)
        self.assertEqual(modelspecs.feature_names, None)

    @patch("openstf.data_classes.model_specifications.ModelSpecificationDataClass")
    @patch("mlflow.search_runs")
    @patch("mlflow.sklearn.load_model")
    def test_serializer_feature_names_attributeerror(
        self, mock_load, mock_search_runs, mock_modelspecs
    ):
        mock_search_runs.return_value = pd.DataFrame(
            data={
                "run_id": [1, 2],
                "artifact_uri": ["path1", "path2"],
                # give wrong feature_name type, something else than a str of a list or dict
                "tags.feature_names": [1, 2],
                "end_time": [
                    datetime.utcnow() - timedelta(days=2),
                    datetime.utcnow() - timedelta(days=3),
                ],
            }
        )
        mock_modelspecs.return_value = self.modelspecs
        type(mock_load.return_value).feature_names = PropertyMock(return_value=None)
        loaded_model, modelspecs = MLflowSerializer(
            trained_models_folder="./test/trained_models"
        ).load_model(307)
        self.assertIsInstance(modelspecs, ModelSpecificationDataClass)
        self.assertEqual(modelspecs.feature_names, None)

    @patch("openstf.data_classes.model_specifications.ModelSpecificationDataClass")
    @patch("mlflow.search_runs")
    @patch("mlflow.sklearn.load_model")
    def test_serializer_feature_names_jsonerror(
        self, mock_load, mock_search_runs, mock_modelspecs
    ):
        mock_search_runs.return_value = pd.DataFrame(
            data={
                "run_id": [1, 2],
                "artifact_uri": ["path1", "path2"],
                # give wrong feature_name type, something else than a str of a list or dict
                "tags.feature_names": ["feature1", "feature1"],
                "end_time": [
                    datetime.utcnow() - timedelta(days=2),
                    datetime.utcnow() - timedelta(days=3),
                ],
            }
        )

        mock_modelspecs.return_value = self.modelspecs
        type(mock_load.return_value).feature_names = PropertyMock(return_value=None)
        loaded_model, modelspecs = MLflowSerializer(
            trained_models_folder="./test/trained_models"
        ).load_model(307)
        self.assertIsInstance(modelspecs, ModelSpecificationDataClass)
        self.assertEqual(modelspecs.feature_names, None)

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
        pj = self.pj
        # set ID to default, so MLflow saves it in a default folder
        pj["id"] = "Default"
        report_mock = MagicMock()
        report_mock.get_metrics.return_value = {"mae", 0.2}
        with self.assertLogs("MLflowSerializer", level="INFO") as captured:

            MLflowSerializer(
                trained_models_folder="./test/trained_models"
            ).save_model(
                model=model, pj=pj, modelspecs=self.modelspecs, report=report_mock
            )
            # The index shifts if logging is added
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
        pj = self.pj
        # set ID to default, so MLflow saves it in a default folder
        pj["id"] = "Default"
        report_mock = MagicMock()
        report_mock.get_metrics.return_value = {"mae", 0.2}
        mock_search.return_value = pd.DataFrame(columns=["run_id"])
        with self.assertLogs("MLflowSerializer", level="INFO") as captured:
            MLflowSerializer(
                trained_models_folder="./test/trained_models"
            ).save_model(
                model=model, pj=pj, modelspecs=self.modelspecs, report=report_mock
            )
            # The index shifts if logging is added
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
        days = MLflowSerializer(
            trained_models_folder="./test/trained_models"
        )._determine_model_age_from_mlflow_run(run)
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
            "MLflowSerializer", level="WARNING"
        ) as captured:
            days = MLflowSerializer(
                trained_models_folder="./test/trained_models"
            )._determine_model_age_from_mlflow_run(run)
        # The index shifts if logging is added
        self.assertRegex(
            captured.records[0].getMessage(),
            "Could not get model age. Returning infinite age!",
        )
        self.assertEqual(days, float("inf"))

    def test_serializer_remove_old_models(self):
        """
        Test if correct number of models are removed when removing old models.
        Test uses 5 previously stored models, then is allowed to keep 2.
        Check if it keeps the 2 most recent models"""

        #####
        # Set up
        local_model_dir = "./test/trained_models/models_for_serializertest"

        ### The code below generates stored models, makes sure there are 4 models
        # We want to test using pre-stored models, since MLflow takes ~6seconds per save_model()
        def store_models(pj, modelspecs, n):
            model_type = "xgb"
            model = ModelCreator.create_model(model_type)
            dummy_report = Report(
                feature_importance_figure=None,
                data_series_figures={},
                metrics={},
                signature=None,
            )
            serializer = MLflowSerializer(local_model_dir)
            for _ in range(n):
                serializer.save_model(model, pj, modelspecs, report=dummy_report)
            return serializer._find_models(pj["id"], n=None)

        # We copy the already stored models to a temp dir and test the functionality from there
        with tempfile.TemporaryDirectory() as temp_model_dir:
            # Copy already stored models to temp dir
            copy_tree(local_model_dir, temp_model_dir)

            serializer = MLflowSerializer(temp_model_dir)
            # Find all stored models
            all_stored_models = serializer._find_models(self.pj["id"], n=None)

            if len(all_stored_models) < 4:
                all_stored_models = store_models(self.pj, self.modelspecs, n=int(4 - len(all_stored_models)))

            # Remove old models
            serializer.remove_old_models(self.pj, max_n_models=2)
            final_stored_models = serializer._find_models(self.pj["id"], n=None)

            # Compare final_stored_models to all_stored_models
            self.assertEqual(len(all_stored_models), 4)
            self.assertEqual(len(final_stored_models), 2)
            # Check if the runs match to the newest two runs
            self.assertDataframeEqual(
                final_stored_models.sort_values(by="end_time", ascending=False),
                all_stored_models.sort_values(by="end_time", ascending=False).iloc[
                    :2, :
                ],
            )

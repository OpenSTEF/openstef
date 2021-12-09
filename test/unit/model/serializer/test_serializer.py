# SPDX-FileCopyrightText: 2017-2021 Contributors to the OpenSTF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

import glob
import tempfile
from datetime import datetime, timedelta
from distutils.dir_util import copy_tree
from test.unit.utils.base import BaseTestCase
from test.unit.utils.data import TestData
from unittest.mock import MagicMock, PropertyMock, patch

import cufflinks  # required for iplot() of pandas
import numpy as np
import pandas as pd

from openstef.data_classes.model_specifications import ModelSpecificationDataClass
from openstef.metrics.reporter import Report
from openstef.model.model_creator import ModelCreator
from openstef.model.serializer import MLflowSerializer


class TestMLflowSerializer(BaseTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.pj, self.modelspecs = TestData.get_prediction_job_and_modelspecs(pid=307)

    # @patch("mlflow.search_runs")
    @patch("mlflow.sklearn.load_model")
    def test_serializer_load_model_artifact_uri_construct(
        self,
        mock_load,
        # mock_search_runs
    ):
        """Explicitly check if the artifact uri is constructed correctly
        when existing model is loaded based on meta.yaml
        This has led to some bugs in the past"""

        loaded_model, _ = MLflowSerializer(
            trained_models_folder="./test/unit/trained_models"
        ).load_model(307)
        # Check model path
        assert (
            loaded_model.path
            == "/*change_this_in_mock_so_it_also_works_remote*\\model/"
        )

    @patch("openstef.data_classes.model_specifications.ModelSpecificationDataClass")
    @patch("mlflow.search_runs")
    @patch("mlflow.sklearn.load_model")
    def test_serializer_load_model_feature_names_keyerror(
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
            trained_models_folder="./test/unit/trained_models"
        ).load_model(307)
        self.assertIsInstance(modelspecs, ModelSpecificationDataClass)
        self.assertEqual(modelspecs.feature_names, None)

    @patch("openstef.data_classes.model_specifications.ModelSpecificationDataClass")
    @patch("mlflow.search_runs")
    @patch("mlflow.sklearn.load_model")
    def test_serializer_load_model_feature_names_attributeerror(
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
            trained_models_folder="./test/unit/trained_models"
        ).load_model(307)
        self.assertIsInstance(modelspecs, ModelSpecificationDataClass)
        self.assertEqual(modelspecs.feature_names, None)

    @patch("openstef.data_classes.model_specifications.ModelSpecificationDataClass")
    @patch("mlflow.search_runs")
    @patch("mlflow.sklearn.load_model")
    def test_serializer_load_model_feature_names_jsonerror(
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
            trained_models_folder="./test/unit/trained_models"
        ).load_model(307)
        self.assertIsInstance(modelspecs, ModelSpecificationDataClass)
        self.assertEqual(modelspecs.feature_names, None)

    @patch("openstef.model.serializer.MLflowSerializer._find_models")
    def test_serializer_load_model_empty_df_raise_lookuperror(self, mock_find_models):
        mock_find_models.return_value = pd.DataFrame()
        self.assertRaises(
            LookupError, MLflowSerializer(trained_models_folder="/").load_model, 307
        )

    @patch("mlflow.sklearn.log_model")
    @patch("mlflow.log_figure")
    @patch("mlflow.log_params")
    @patch("mlflow.log_metrics")
    @patch("mlflow.set_tag")
    @patch("mlflow.search_runs")
    def test_serializer_save_model(
        self,
        mock_search_runs,
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
        pj["id"] = "307"

        # Build reporter mock, including html figures so we can test if they are stored correctly
        report_mock = MagicMock()
        report_mock.get_metrics.return_value = {"mae", 0.2}
        # define dummy plot
        dummy_fig = pd.DataFrame(index=[1, 2], data={"dummy": [1, 2]}).iplot(
            asFigure=True
        )
        report_mock.feature_importance_figure = dummy_fig
        report_mock.data_series_figures = {
            "Predictor1": dummy_fig,
            "Predictor2": dummy_fig,
        }

        # Use tempdir structure to test if model reports are stored correctly
        with tempfile.TemporaryDirectory() as trained_models_folder:
            with self.assertLogs("MLflowSerializer", level="INFO") as captured:
                MLflowSerializer(
                    trained_models_folder=trained_models_folder
                ).save_model(
                    model=model, pj=pj, modelspecs=self.modelspecs, report=report_mock
                )
                # The index shifts if logging is added
                self.assertEqual(
                    captured.records[1].msg["event"], "Model saved with MLflow"
                )
            # Check if reports are stored in correct place
            report_files = glob.glob(f"{trained_models_folder}/{self.pj['id']}/*.html")
            self.assertEqual(len(report_files), 3)

    @patch("mlflow.sklearn.log_model")
    @patch("mlflow.log_figure")
    @patch("mlflow.log_params")
    @patch("mlflow.log_metrics")
    @patch("mlflow.set_tag")
    @patch("mlflow.search_runs")
    def test_serializer_save_model_no_previous(
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
                trained_models_folder="./test/unit/trained_models"
            ).save_model(
                model=model, pj=pj, modelspecs=self.modelspecs, report=report_mock
            )
            # The index shifts if logging is added
            self.assertRegex(
                captured.records[0].getMessage(), "No previous model found in MLflow"
            )

    @patch("openstef.model.serializer.MLflowSerializer._find_models")
    def test_serializer_get_model_age_no_hyperparameter_optimization(
        self, mock_find_models
    ):
        models_df = pd.DataFrame(
            data={
                "run_id": [1],
                "artifact_uri": ["path1"],
                "end_time": [datetime.utcnow() - timedelta(days=2)],
            }
        )
        mock_find_models.return_value = models_df
        days = MLflowSerializer(
            trained_models_folder="./test/unit/trained_models"
        ).get_model_age(307, hyperparameter_optimization_only=False)
        self.assertEqual(days, 2)

    @patch("openstef.model.serializer.MLflowSerializer._find_models")
    def test_serializer_get_model_age_hyperparameter_optimization(
        self, mock_find_models
    ):
        models_df = pd.DataFrame(
            data={
                "run_id": [1, 2],
                "artifact_uri": ["path1", "path2"],
                "end_time": [datetime.utcnow() - timedelta(days=8), datetime.utcnow()],
            }
        )
        mock_find_models.return_value = models_df
        days = MLflowSerializer(
            trained_models_folder="./test/unit/trained_models"
        ).get_model_age(307, hyperparameter_optimization_only=True)
        self.assertGreater(days, 7)
        self.assertEqual(days, 8)

    @patch("openstef.model.serializer.MLflowSerializer._find_models")
    def test_serializer_get_model_age_empty_df(self, mock_find_models):
        models_df = pd.DataFrame()
        mock_find_models.return_value = models_df
        days = MLflowSerializer(
            trained_models_folder="./test/unit/trained_models"
        ).get_model_age(307, hyperparameter_optimization_only=True)
        self.assertGreater(days, 7)
        self.assertEqual(days, np.inf)

    def test_serializer_determine_model_age_from_MLflow_run(self):
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
            trained_models_folder="./test/unit/trained_models"
        )._determine_model_age_from_mlflow_run(run)
        self.assertGreater(days, 7)

    def test_serializer_determine_model_age_from_MLflow_run_exception(self):
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
        with self.assertLogs("MLflowSerializer", level="WARNING") as captured:
            days = MLflowSerializer(
                trained_models_folder="./test/unit/trained_models"
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
        # Set up
        local_model_dir = "./test/unit/trained_models/models_for_serializertest"

        # Run the code below once, to generate stored models
        # We want to test using pre-stored models, since it takes ~6s per save_model()
        # If you want to store new models, run the lines below:
        if False:  # set to true if we want to generate models
            model_type = "xgb"
            model = ModelCreator.create_model(model_type)
            dummy_report = Report(
                feature_importance_figure=None,
                data_series_figures={},
                metrics={},
                signature=None,
            )
            serializer = MLflowSerializer(local_model_dir)
            for _ in range(4):
                serializer.save_model(
                    model, self.pj, self.modelspecs, report=dummy_report
                )

        # We copy the stored models to a temp dir and test the functionality from there
        with tempfile.TemporaryDirectory() as temp_model_dir:
            # Copy already stored models to temp dir
            copy_tree(local_model_dir, temp_model_dir)

            serializer = MLflowSerializer(temp_model_dir)
            # Find all stored models
            all_stored_models = serializer._find_models(self.pj["id"])

            # Remove old models
            serializer.remove_old_models(self.pj, max_n_models=2)

            # Check which models are left
            final_stored_models = serializer._find_models(self.pj["id"])
            # Compare final_stored_models to all_stored_models
            self.assertEqual(
                len(all_stored_models),
                4,
                f"we expect 4 models at the start- (now {len(all_stored_models)}), "
                f"please remove runs (manually) or add runs with MAKE_RUNS == TRUE ",
            )
            self.assertEqual(len(final_stored_models), 2)
            # Check if the runs match to the oldest two runs
            self.assertDataframeEqual(
                final_stored_models.sort_values(by="end_time", ascending=False),
                all_stored_models.sort_values(by="end_time", ascending=False).iloc[
                    :2, :
                ],
            )

            # Check if models are removed from disk
            model_dirs = glob.glob(f"{temp_model_dir}/mlruns/1/*/", recursive=True)
            self.assertEqual(len(model_dirs), 2)

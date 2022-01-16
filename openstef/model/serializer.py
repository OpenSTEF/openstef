# SPDX-FileCopyrightText: 2017-2021 Contributors to the OpenSTF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
import json
import os
import shutil
import structlog
from datetime import datetime
from json import JSONDecodeError
from pathlib import Path
from typing import Optional, Tuple, Union
from urllib.parse import unquote, urlparse

import mlflow
import numpy as np
import pandas as pd
from matplotlib import figure
from mlflow.exceptions import MlflowException
from openstef_dbc.services.prediction_job import PredictionJobDataClass
from plotly import graph_objects

from openstef.data_classes.model_specifications import ModelSpecificationDataClass
from openstef.metrics.reporter import Report, Reporter
from openstef.model.regressors.regressor import OpenstfRegressor

MODEL_FILENAME = "model.joblib"
FOLDER_DATETIME_FORMAT = "%Y%m%d%H%M%S"
MODEL_ID_SEP = "-"
MAX_N_MODELS = 10  # Number of models per experiment allowed in model registry
E_MSG = "feature_names couldn't be loaded, using None"


class MLflowSerializer:
    def __init__(
        self, trained_models_folder: Union[Path, str], mlflow_tracking_uri: str = None
    ):
        # TODO: remove trained_models_folder once users have gone to mlflow_tracking_uri
        self.logger = structlog.get_logger(self.__class__.__name__)
        self.trained_models_folder = trained_models_folder
        if mlflow_tracking_uri:  # setup distributed mlflow from uri
            self.mlflow_folder = mlflow_tracking_uri
        else:  # setup local mlflow
            path = os.path.abspath(f"{trained_models_folder}/mlruns/")
            self.mlflow_folder = Path(path).as_uri()
        mlflow.set_tracking_uri(self.mlflow_folder)
        self.web_volume = Path(f"{trained_models_folder}")
        self.logger.debug(f"MLflow path at init= {self.mlflow_folder}")
        self.experiment_id = None

    def save_model(
        self,
        model: OpenstfRegressor,
        pj: PredictionJobDataClass,
        modelspecs: ModelSpecificationDataClass,
        report: Report,
        phase: str = "training",
        save_report_to_disk: bool = True,
        **kwargs,
    ) -> None:
        """Save sklearn compatible model to persistent storage with MLflow.

            Either a pid or a model_id should be given. If a pid is given the model_id
            will be generated.

        Args:
            model (OpenstfRegressor): Trained sklearn compatible model object.
            pj (PredictionJobDataClass): Prediction job.
            modelspecs (ModelSpecificationDataClass): Dataclass containing model specifications
            report (Report): Report object.
            phase (str): Where does the model come from, default is "training"
            save_report_to_disk (bool): Do we want to save report to disk besides saving reports on MLFlow.
            **kwargs: Extra information to be logged with mlflow, this can add the extra modelspecs

        """

        models_df = self._find_models(pj["id"], n=1)  # returns latest model
        if not models_df.empty:
            prev_run_id = models_df["run_id"][0]  # Use [0] to only get latest run id
        else:
            self.logger.info("No previous model found in MLflow", pid=pj["id"])
            prev_run_id = None
        with mlflow.start_run(run_name=pj["model"]):
            self._log_model_with_mlflow(
                pj, modelspecs, model, report, phase, prev_run_id, **kwargs
            )
            self._log_figure_with_mlflow(report)
        self.logger.debug(f"MLflow path after saving= {self.mlflow_folder}")

        if save_report_to_disk:
            # Report is already stored in MLFlow.
            # This part also optionally stores report files per model on disk.
            # Easy for web visualisation, e.g. through grafana
            location = os.path.join(self.web_volume, f'{pj["id"]}')
            Reporter.write_report_to_disk(report, location=location)
            self.logger.info(f"Stored report to disk: {location}")

    def load_model(
        self,
        pid: Union[str, int],
    ) -> Tuple[OpenstfRegressor, ModelSpecificationDataClass]:
        """Load sklearn compatible model from persistent storage.

            If a pid is given the most recent model for that pid will be loaded.

        Args:
            pid (int): prediction job id

        Raises:
            AttributeError: when there is no experiment with pid in MLflow
            LookupError: when there is no model in MLflow
            OSError: When directory doesn't exist
            MlflowException: When MLflow is not able to log

        Returns:
            OpenstfRegressor: Loaded model
            ModelSpecificationDataClass: model specifications
        """

        try:
            # return the latest run of the model
            models_df = self._find_models(pid, n=1)
            if not models_df.empty:
                latest_run = models_df.iloc[0]  # Use .iloc[0] to only get latest run
            else:
                self.logger.info("No previous model found in MLflow", pid=pid)
                raise LookupError(
                    f"Model couldn't be found for pid {pid}. First train a model!"
                )

            # Get model uri
            model_uri = _get_model_uri(latest_run.artifact_uri)
            loaded_model = mlflow.sklearn.load_model(model_uri)

            # Add model age to model object
            loaded_model.age = self._determine_model_age_from_mlflow_run(latest_run)

            # get model specifications
            modelspecs = self._get_model_specs(pid, loaded_model, latest_run)

            # Path without file:///
            loaded_model.path = unquote(urlparse(model_uri).path)
            self.logger.info("Model successfully loaded with MLflow")
            return loaded_model, modelspecs
        # Catch possible errors
        except (AttributeError, MlflowException, OSError) as e:
            self.logger.error(
                "Couldn't load model",
                pid=pid,
                error=e,
            )
            raise AttributeError(
                "Model couldn't be found or doesn't exist. First train a model!"
            )

    def get_model_age(
        self, pid: Union[int, str], hyperparameter_optimization_only: bool = False
    ) -> int:

        filter_string = "attribute.status = 'FINISHED'"
        if hyperparameter_optimization_only:
            filter_string += " AND tags.phase = 'Hyperparameter_opt'"

        # get models
        models_df = self._find_models(pid, n=1, filter_string=filter_string)

        if not models_df.empty:
            # get age of model
            run = models_df.iloc[0]
            return self._determine_model_age_from_mlflow_run(run)
        else:
            self.logger.info("No model found returning infinite model age!")
            return np.inf

    def _get_model_specs(
        self,
        pid: Union[int, str],
        loaded_model: OpenstfRegressor,
        latest_run: pd.Series,
    ) -> ModelSpecificationDataClass:
        """get model specifications from a model

        Args:
            pid (int): prediction job id
            loaded_model (OpenstfRegressor): previously trained model
            latest_run (pd.Series): last MLflow run

        Returns:
            ModelSpecificationDataclass: model specification to use for this model
        """
        # create basic modelspecs
        modelspecs = ModelSpecificationDataClass(id=pid)

        # get the parameters from the old model, we insert these later into the new model
        modelspecs.hyper_params = loaded_model.get_params()

        # get used feature names else use all feature names
        modelspecs.feature_names = self._get_feature_names(
            pid, latest_run, modelspecs, loaded_model
        )

        return modelspecs

    def _set_experiment(self, pid: Union[int, str]) -> str:
        """Setup experiment for MLFLow

        Args:
            pid (int): Prediction job id

        Returns:
            int: The experiment id of the prediction job

        """
        mlflow.set_experiment(str(pid))
        return mlflow.get_experiment_by_name(str(pid)).experiment_id

    def _find_models(
        self,
        pid: Union[int, str],
        n: Optional[int] = None,
        filter_string: str = None,
    ) -> pd.DataFrame:
        """
        Finds trained models for specific pid sorted by age in descending order.

        Args:
            pid (PredictionJobDataClass): Prediction job id
            n (int): return the n latest models, default = None
            filter_string (str): filter runs (e.g. "attribute.status = 'FINISHED'")


        Returns:
            pd.DataFrame: models_df (this dataframe can have 0, 1 or multiple rows)
        """
        self.experiment_id = self._set_experiment(pid)

        if filter_string is None:
            filter_string = "attribute.status = 'FINISHED'"

        if isinstance(n, int):
            models_df = mlflow.search_runs(
                self.experiment_id,
                filter_string=filter_string,
                max_results=n,
            )

            if n == 1 and len(models_df) > 0:
                models_df = models_df.iloc[:1]  # filter on first row of dataframe
        else:
            models_df = mlflow.search_runs(
                self.experiment_id,
                filter_string=filter_string,
            )
        return models_df

    def _determine_model_age_from_mlflow_run(self, run: pd.Series) -> Union[int, float]:
        """Determines how many days ago a model is trained from the mlflow run

        Args:
            run (mlfow run): run containing the information about the trained model

        Returns:
            model_age_days (int): age of the model
        """
        try:
            model_datetime = run.end_time.to_pydatetime()
            model_datetime = model_datetime.replace(tzinfo=None)
            model_age_days = (datetime.utcnow() - model_datetime).days
        except Exception as e:
            self.logger.warning(
                "Could not get model age. Returning infinite age!", exception=e
            )
            return np.inf  # Return fallback age
        return model_age_days

    def _log_model_with_mlflow(
        self,
        pj: PredictionJobDataClass,
        modelspecs: ModelSpecificationDataClass,
        model: OpenstfRegressor,
        report: Report,
        phase: str,
        prev_run_id: str,
        **kwargs,
    ) -> None:
        """Log model with MLflow

        Args:
            pj (PredictionJobDataClass): Prediction job
            model (OpenstfRegressor): Model to be logged
            report (Report): report where the info is stored
            phase (str): Origin of the model (Training or Hyperparameter_opt)
            prev_run_id (str): Run-id of the previous run in this prediction job
            **kwargs: Extra information to be logged with mlflow

        """

        # Set tags to the run, can be used to filter on the UI
        mlflow.set_tag("run_id", mlflow.active_run().info.run_id)
        mlflow.set_tag("phase", phase)
        mlflow.set_tag("Previous_version_id", prev_run_id)
        mlflow.set_tag("model_type", pj["model"])
        mlflow.set_tag("prediction_job", pj["id"])

        # add modelspecs attributes except hyper_params

        # save feature names and target to MLflow, assume target is the first column
        mlflow.set_tag("feature_names", modelspecs.feature_names[1:])
        mlflow.set_tag("target", modelspecs.feature_names[0])

        # Add metrics to the run
        mlflow.log_metrics(report.metrics)
        # Add the used parameters to the run + the params from the prediction job
        modelspecs.hyper_params.update(model.get_params())
        mlflow.log_params(modelspecs.hyper_params)

        # Process args
        for key, value in kwargs.items():
            if isinstance(value, dict):
                mlflow.log_dict(value, f"{key}.json")
            elif isinstance(value, str) or isinstance(value, int):
                mlflow.set_tag(key, value)
            elif isinstance(value, graph_objects.Figure):
                mlflow.log_figure(value, f"figures/{key}.html")
            elif isinstance(value, figure.Figure):
                mlflow.log_figure(value, f"figures/{key}.png")
            else:
                self.logger.warning(
                    f"Couldn't log {key}, {type(key)} not supported", pid=pj["id"]
                )

        # Log the model to the run
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=report.signature,
        )
        self.logger.info("Model saved with MLflow", pid=pj["id"])

    def _log_figure_with_mlflow(self, report) -> None:
        """Log model with MLflow

        Args:
            report (Report): report where the info is stored

        """
        # log reports/figures in the artifact folder

        if report.feature_importance_figure is not None:
            mlflow.log_figure(
                report.feature_importance_figure, "figures/weight_plot.html"
            )

        for key, fig in report.data_series_figures.items():
            mlflow.log_figure(fig, f"figures/{key}.html")
        self.logger.info(f"logged figures to MLflow")

    def _find_all_models(self, pj: PredictionJobDataClass):
        experiment_id = self._set_experiment(pj["id"])
        prev_runs = mlflow.search_runs(
            experiment_id,
            filter_string=" attribute.status = 'FINISHED' AND tags.mlflow.runName = '{}'".format(
                pj["model"]
            ),
        )
        return prev_runs

    def remove_old_models(
        self, pj: PredictionJobDataClass, max_n_models: int = MAX_N_MODELS
    ):
        """Remove old models for the experiment defined by PJ.
        A maximum of 'max_n_models' is allowed.
        Note that the current implementation only works if the Storage backend is used
        This functionality is not incorporated in MLFlow natively
        See also: https://github.com/mlflow/mlflow/issues/2152"""
        if max_n_models < 1:
            raise ValueError(
                f"MAX_N_MODELS should be greater than 1! Received: {max_n_models}"
            )

        prev_runs = self._find_all_models(pj)

        if len(prev_runs) > max_n_models:
            self.logger.debug(
                f"Going to delete old models. {len(prev_runs)}>{max_n_models}"
            )
            # Find run_ids of oldest runs
            runs_to_remove = prev_runs.sort_values(by="end_time", ascending=False).loc[
                max_n_models:, :
            ]
            for _, run in runs_to_remove.iterrows():
                artifact_location = os.path.join(
                    self.trained_models_folder,
                    f"mlruns/{run.experiment_id}/{run.run_id}",
                )
                self.logger.debug(
                    f"Going to remove run {run.run_id}, from {run.end_time}."
                    f" Artifact location: {artifact_location}"
                )
                mlflow.delete_run(run.run_id)
                # Also remove artifact from disk.
                # mlflow.delete_run only marks it as deleted but does not delete it by itself
                try:
                    shutil.rmtree(artifact_location)
                except Exception as e:
                    self.logger.info(f"Failed removing artifacts: {e}")

                self.logger.debug("Removed run")

    def _get_feature_names(
        self,
        pid: Union[int, str],
        latest_run: pd.Series,
        modelspecs: ModelSpecificationDataClass,
        loaded_model: OpenstfRegressor,
    ) -> Optional[list]:
        """Get the feature_names from MLflow or the old model

        Args:
            pid: prediction job id
            latest_run: pandas series of the last MLflow run
            modelspecs: model specification
            loaded_model: previous model

        Returns:
            list: feature names to use
        """
        try:
            modelspecs.feature_names = json.loads(
                latest_run["tags.feature_names"].replace("'", '"')
            )

        except KeyError:
            self.logger.warning(
                E_MSG,
                pid=pid,
                error="tags.feature_names, doesn't exist in run",
            )
        except AttributeError:
            self.logger.warning(
                E_MSG,
                pid=pid,
                error="tags.feature_names, needs to be a string",
            )
        except JSONDecodeError:
            self.logger.warning(
                E_MSG,
                pid=pid,
                error="tags.feature_names, needs to be a string of a list",
            )

        # todo: this code should become absolute after a few runs
        # if feature names is non see if we can retrieve them from the old model
        if modelspecs.feature_names is None:
            try:
                if loaded_model.feature_names is not None:
                    modelspecs.feature_names = loaded_model.feature_names
                    self.logger.info(
                        "feature_names retrieved from old model with an attribute",
                        pid=pid,
                    )
            except AttributeError:
                self.logger.warning(
                    "feature_names not an attribute of the old model, using None ",
                    pid=pid,
                )
        return modelspecs.feature_names


def _get_model_uri(artifact_uri: str) -> str:
    """Set model uri based on latest run.
    Note that this function is primarily useful
    so it can be mocked during unit tests"""
    return os.path.join(artifact_uri, "model/")

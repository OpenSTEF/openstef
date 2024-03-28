# SPDX-FileCopyrightText: 2017-2023 Contributors to the OpenSTEF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
import json
import logging
import os
import shutil
from datetime import datetime, UTC
from json import JSONDecodeError
from typing import Optional, Union
from urllib.parse import unquote, urlparse

import mlflow
import numpy as np
import pandas as pd
import structlog
from mlflow.exceptions import MlflowException
from mlflow.store.artifact.artifact_repository_registry import get_artifact_repository
from xgboost import XGBModel  # Temporary for backward compatibility

from openstef.data_classes.model_specifications import ModelSpecificationDataClass
from openstef.metrics.reporter import Report
from openstef.model.regressors.regressor import OpenstfRegressor
from openstef.settings import Settings


class MLflowSerializer:
    def __init__(self, mlflow_tracking_uri: str):
        structlog.configure(
            wrapper_class=structlog.make_filtering_bound_logger(
                logging.getLevelName(Settings.log_level)
            )
        )
        self.logger = structlog.get_logger(self.__class__.__name__)
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        self.logger.debug(f"MLflow tracking uri at init= {mlflow_tracking_uri}")
        self.experiment_name_prefix = (
            os.environ["DATABRICKS_WORKSPACE_PATH"]
            if "DATABRICKS_WORKSPACE_PATH" in os.environ
            else ""
        )

    def save_model(
        self,
        model: OpenstfRegressor,
        experiment_name: str,
        model_type: str,
        model_specs: ModelSpecificationDataClass,
        report: Report,
        phase: str = "training",
        **kwargs,
    ) -> None:
        """Save sklearn compatible model to MLFlow."""
        mlflow.set_experiment(
            experiment_name=self.experiment_name_prefix + experiment_name
        )
        with mlflow.start_run(run_name=experiment_name):
            self._log_model_with_mlflow(
                model=model,
                experiment_name=experiment_name,
                model_type=model_type,
                model_specs=model_specs,
                report=report,
                phase=phase,
                **kwargs,
            )
            self._log_figures_with_mlflow(report)

    def _log_model_with_mlflow(
        self,
        model: OpenstfRegressor,
        experiment_name: str,
        model_type: str,
        model_specs: ModelSpecificationDataClass,
        report: Report,
        phase: str,
        **kwargs,
    ) -> None:
        """Log model with MLflow.

        Note: **kwargs has extra information to be logged with mlflow

        """
        # Get previous run id
        models_df = self._find_models(
            self.experiment_name_prefix + experiment_name, max_results=1
        )  # returns latest model
        if not models_df.empty:
            previous_run_id = models_df["run_id"][
                0
            ]  # Use [0] to only get latest run id
        else:
            self.logger.info(
                "No previous model found in MLflow", experiment_name=experiment_name
            )
            previous_run_id = None

        # Set tags to the run, can be used to filter on the UI
        mlflow.set_tag("run_id", mlflow.active_run().info.run_id)
        mlflow.set_tag("phase", phase)  # phase can be Training or Hyperparameter_opt
        mlflow.set_tag("Previous_version_id", previous_run_id)
        mlflow.set_tag("model_type", model_type)
        mlflow.set_tag("prediction_job", experiment_name)

        # Add feature names, target, feature modules, metrics and params to the run
        mlflow.set_tag(
            "feature_names", model_specs.feature_names[1:]
        )  # feature names are 1+ columns
        mlflow.set_tag("target", model_specs.feature_names[0])  # target is first column
        mlflow.set_tag("feature_modules", model_specs.feature_modules)
        mlflow.log_metrics(report.metrics)
        model_specs.hyper_params.update(model.get_params())
        # TODO: Remove this hardcoded hyper params fix with loop after fix by mlflow
        # https://github.com/mlflow/mlflow/issues/6384
        for key, value in model_specs.hyper_params.items():
            if value == "":
                model_specs.hyper_params[key] = " "
        mlflow.log_params(model_specs.hyper_params)

        # Process args
        for key, value in kwargs.items():
            if isinstance(value, dict):
                mlflow.log_dict(value, f"{key}.json")
            elif isinstance(value, str) or isinstance(value, int):
                mlflow.set_tag(key, value)
            else:
                self.logger.warning(
                    f"Couldn't log {key}, {type(key)} not supported",
                    experiment_name=experiment_name,
                )

        # Log the model to the run. Signature describes model input and output scheme
        mlflow.sklearn.log_model(
            sk_model=model, artifact_path="model", signature=report.signature
        )
        self.logger.info("Model saved with MLflow", experiment_name=experiment_name)

    def _log_figures_with_mlflow(self, report) -> None:
        """Log figures with MLflow in the artifact folder."""
        if report.feature_importance_figure is not None:
            mlflow.log_figure(
                report.feature_importance_figure, "figures/weight_plot.html"
            )
        for key, figure in report.data_series_figures.items():
            mlflow.log_figure(figure, f"figures/{key}.html")
        self.logger.info("Logged figures to MLflow.")

    def load_model(
        self,
        experiment_name: str,
    ) -> tuple[OpenstfRegressor, ModelSpecificationDataClass]:
        """Load sklearn compatible model from MLFlow.

        Args:
            experiment_name: Name of the experiment, often the id of the predition job.

        Raises:
            LookupError: If model is not found in MLflow.

        """
        try:
            models_df = self._find_models(
                self.experiment_name_prefix + experiment_name, max_results=1
            )  # return the latest finished run of the model
            if not models_df.empty:
                latest_run = models_df.iloc[0]  # Use .iloc[0] to only get latest run
            else:
                raise LookupError("Model not found. First train a model!")
            model_uri = self._get_model_uri(latest_run.artifact_uri)
            loaded_model = mlflow.sklearn.load_model(model_uri)
            loaded_model.age = self._determine_model_age_from_mlflow_run(latest_run)
            model_specs = self._get_model_specs(
                experiment_name, loaded_model, latest_run
            )
            loaded_model.path = unquote(
                urlparse(model_uri).path
            )  # Path without file:///
            self.logger.info("Model successfully loaded with MLflow")
            return loaded_model, model_specs
        except (AttributeError, MlflowException, OSError) as exception:
            raise LookupError("Model not found. First train a model!") from exception

    def get_model_age(
        self, experiment_name: str, hyperparameter_optimization_only: bool = False
    ) -> int:
        """Get model age of most recent model.

        Args:
            experiment_name: Name of the experiment, often the id of the predition job.
            hyperparameter_optimization_only: Set to true if only hyperparameters optimaisation events should be considered.

        """
        filter_string = "attribute.status = 'FINISHED'"
        if hyperparameter_optimization_only:
            filter_string += " AND tags.phase = 'Hyperparameter_opt'"
        models_df = self._find_models(
            self.experiment_name_prefix + experiment_name,
            max_results=1,
            filter_string=filter_string,
        )
        if not models_df.empty:
            run = models_df.iloc[0]  # Use .iloc[0] to only get latest run
            return self._determine_model_age_from_mlflow_run(run)
        else:
            self.logger.info("No model found returning infinite model age!")
            return np.inf

    def _find_models(
        self,
        experiment_name: str,
        max_results: Optional[int] = 100,
        filter_string: str = "attribute.status = 'FINISHED'",
    ) -> pd.DataFrame:
        """Finds trained models for specific experiment_name sorted by age in descending order."""
        models_df = mlflow.search_runs(
            experiment_names=[experiment_name],
            max_results=max_results,
            filter_string=filter_string,
        )
        return models_df

    def _get_model_specs(
        self,
        experiment_name: str,
        loaded_model: OpenstfRegressor,
        latest_run: pd.Series,
    ) -> ModelSpecificationDataClass:
        """Get model specifications from existing model."""
        model_specs = ModelSpecificationDataClass(id=experiment_name)

        # Temporary fix for update of xgboost
        # new version requires some attributes that the old (stored) models don't have yet
        # see: https://stackoverflow.com/questions/71912084/attributeerror-xgbmodel-object-has-no-attribute-callbacks
        new_attrs = [
            "grow_policy",
            "max_bin",
            "eval_metric",
            "callbacks",
            "early_stopping_rounds",
            "max_cat_to_onehot",
            "max_leaves",
            "sampling_method",
        ]

        manual_additional_attrs = [
            "enable_categorical",
            "predictor",
        ]  # these ones are not mentioned in the stackoverflow post
        automatic_additional_attrs = [
            x
            for x in XGBModel._get_param_names()
            if x
            not in new_attrs + manual_additional_attrs + loaded_model._get_param_names()
        ]

        for attr in new_attrs + manual_additional_attrs + automatic_additional_attrs:
            setattr(loaded_model, attr, None)

        # This one is new is should be set to a specific value (https://xgboost.readthedocs.io/en/latest/python/python_api.html#module-xgboost.training)
        setattr(loaded_model, "missing", np.nan)
        setattr(loaded_model, "n_estimators", 100)

        # End temporary fix

        # get the parameters from old model, we insert these later into new model
        model_specs.hyper_params = loaded_model.get_params()
        # TODO: Remove this hardcoded hyper params fix with loop after fix by mlflow
        # https://github.com/mlflow/mlflow/issues/6384
        for key, value in model_specs.hyper_params.items():
            if value == " ":
                model_specs.hyper_params[key] = ""
        # get used feature names else use all feature names
        model_specs.feature_names = self._get_feature_names(
            experiment_name, latest_run, model_specs, loaded_model
        )
        # get feature_modules
        model_specs.feature_modules = self._get_feature_modules(
            experiment_name, latest_run, model_specs, loaded_model
        )
        return model_specs

    def _determine_model_age_from_mlflow_run(self, run: pd.Series) -> Union[int, float]:
        """Determines how many days ago a model is trained from the mlflow run."""
        try:
            model_datetime = run.end_time.to_pydatetime()
            model_age_days = (datetime.now(tz=UTC) - model_datetime).days
        except Exception as e:
            self.logger.warning(
                "Could not get model age. Returning infinite age!", exception=str(e)
            )
            return np.inf  # Return fallback age
        return model_age_days

    def remove_old_models(
        self,
        experiment_name: str,
        max_n_models: int = 10,
    ):
        """Remove old models per experiment."""
        if max_n_models < 1:
            raise ValueError(
                f"Max models to keep should be greater than 1! Received: {max_n_models}"
            )
        previous_runs = self._find_models(
            experiment_name=self.experiment_name_prefix + experiment_name
        )
        if len(previous_runs) > max_n_models:
            self.logger.debug(
                f"Going to delete old models. {len(previous_runs)} > {max_n_models}"
            )
            # Find run_ids of oldest runs
            runs_to_remove = previous_runs.sort_values(
                by="end_time", ascending=False
            ).loc[max_n_models:, :]
            for _, run in runs_to_remove.iterrows():
                self.logger.debug(
                    f"Going to remove run {run.run_id}, from {run.end_time}."
                )
                mlflow.delete_run(run.run_id)
                self.logger.debug("Removed run")

                # mlflow.delete_run marks it as deleted but does not delete it by itself
                # Remove artifacts to save disk space
                try:
                    repository = get_artifact_repository(
                        mlflow.get_run(run.run_id).info.artifact_uri
                    )
                    repository.delete_artifacts()
                    self.logger.debug("Removed artifacts")
                except Exception as e:
                    self.logger.info(f"Failed removing artifacts: {e}")

    def _get_feature_names(
        self,
        experiment_name: str,
        latest_run: pd.Series,
        model_specs: ModelSpecificationDataClass,
        loaded_model: OpenstfRegressor,
    ) -> list:
        """Get the feature_names from MLflow or the old model."""
        error_message = "feature_names not loaded and using None, because it"
        try:
            model_specs.feature_names = json.loads(
                latest_run["tags.feature_names"].replace("'", '"')
            )
        except KeyError:
            self.logger.warning(
                f"{error_message} did not exist in run",
                experiment_name=experiment_name,
            )
        except AttributeError:
            self.logger.warning(
                f"{error_message} needs to be a string",
                experiment_name=experiment_name,
            )
        except JSONDecodeError:
            self.logger.warning(
                f"{error_message} needs to be a string of a list",
                experiment_name=experiment_name,
            )

        # if feature names is none, see if we can retrieve them from the old model
        if model_specs.feature_names is None:
            try:
                if loaded_model.feature_names is not None:
                    model_specs.feature_names = loaded_model.feature_names
                    self.logger.info(
                        "feature_names retrieved from old model with an attribute",
                        experiment_name=experiment_name,
                    )
            except AttributeError:
                self.logger.warning(
                    "feature_names not an attribute of the old model, using None ",
                    experiment_name=experiment_name,
                )
        return model_specs.feature_names

    def _get_feature_modules(
        self,
        experiment_name: str,
        latest_run: pd.Series,
        model_specs: ModelSpecificationDataClass,
        loaded_model: OpenstfRegressor,
    ) -> list:
        """Get the feature_modules from MLflow or the old model."""
        error_message = "feature_modules not loaded and using None, because it"
        try:
            model_specs.feature_modules = json.loads(
                latest_run["tags.feature_modules"].replace("'", '"')
            )

        except KeyError:
            self.logger.warning(
                f"{error_message} did not exist in run",
                experiment_name=experiment_name,
            )
        except AttributeError:
            self.logger.warning(
                f"{error_message} needs to be a string",
                experiment_name=experiment_name,
            )
        except JSONDecodeError:
            self.logger.warning(
                f"{error_message} needs to be a string of a list",
                experiment_name=experiment_name,
            )

        # if feature modules is none, see if we can retrieve them from the old model
        if not model_specs.feature_modules:
            try:
                if loaded_model.feature_modules:
                    model_specs.feature_modules = loaded_model.feature_modules
                    self.logger.info(
                        "feature_modules retrieved from old model with an attribute",
                        experiment_name=experiment_name,
                    )
            except AttributeError:
                self.logger.warning(
                    "feature_modules not an attribute of the old model, using None ",
                    experiment_name=experiment_name,
                )
        return model_specs.feature_modules

    def _get_model_uri(self, artifact_uri: str) -> str:
        """Set model uri based on latest run.

        Note: this function helps to mock during unit tests

        """
        return os.path.join(artifact_uri, "model/")

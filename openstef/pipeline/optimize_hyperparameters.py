# SPDX-FileCopyrightText: 2017-2023 Contributors to the OpenSTEF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
import os
from typing import Any, Union

import optuna
import pandas as pd
import structlog

from openstef.data_classes.model_specifications import ModelSpecificationDataClass
from openstef.data_classes.prediction_job import PredictionJobDataClass
from openstef.exceptions import (
    InputDataInsufficientError,
    InputDataWrongColumnOrderError,
)
from openstef.feature_engineering.feature_applicator import TrainFeatureApplicator
from openstef.metrics.reporter import Report, Reporter
from openstef.model.model_creator import ModelCreator
from openstef.model.objective import RegressorObjective
from openstef.model.objective_creator import ObjectiveCreator
from openstef.model.regressors.regressor import OpenstfRegressor
from openstef.model.serializer import MLflowSerializer
from openstef.pipeline.train_model import (
    DEFAULT_TRAIN_HORIZONS_HOURS,
    train_model_pipeline_core,
)
from openstef.validation import validation
from openstef.model_selection.model_selection import split_data_train_validation_test

optuna.logging.enable_propagation()  # Propagate logs to the root logger.
optuna.logging.disable_default_handler()  # Stop showing logs in sys.stderr.

logger = structlog.get_logger(__name__)

# See https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.Study.html#optuna.study.Study.optimize
N_TRIALS: int = 100  # The number of trials.
TIMEOUT: int = 600  # Stop study after the given number of second(s).


def optimize_hyperparameters_pipeline(
    pj: PredictionJobDataClass,
    input_data: pd.DataFrame,
    mlflow_tracking_uri: str,
    artifact_folder: str,
    n_trials: int = N_TRIALS,
) -> dict:
    """Optimize hyperparameters pipeline.

    Expected prediction job key's: "name", "model"

    Args:
        pj: Prediction job
        input_data: Raw training input data
        mlflow_tracking_uri: Path/Uri to mlflow service
        artifact_folder: Path where artifacts, such as trained models, are stored
        horizons: horizons for feature engineering.
        n_trials: The number of trials. Defaults to N_TRIALS.

    Raises:
        ValueError: If the input_date is insufficient.

    Returns:
        Optimized hyperparameters.

    """
    if pj.train_horizons_minutes is None:
        horizons = DEFAULT_TRAIN_HORIZONS_HOURS
    else:
        horizons = [
            horizon_minutes / 60 for horizon_minutes in pj.train_horizons_minutes
        ]
    (
        best_model,
        model_specs,
        report,
        trials,
        best_trial_number,
        best_params,
    ) = optimize_hyperparameters_pipeline_core(pj, input_data, horizons, n_trials)

    # Create serializer
    serializer = MLflowSerializer(mlflow_tracking_uri=mlflow_tracking_uri)

    # Save model, optimization results and report
    serializer.save_model(
        model=best_model,
        experiment_name=str(pj["id"]),
        model_type=pj["model"],
        model_specs=model_specs,
        report=report,
        phase="Hyperparameter_opt",
        trials=trials,
        trial_number=best_trial_number,
    )
    if artifact_folder:
        report_folder = os.path.join(artifact_folder, str(pj["id"]))
        Reporter.write_report_to_disk(report=report, report_folder=report_folder)
    return best_params


def optimize_hyperparameters_pipeline_core(
    pj: PredictionJobDataClass,
    input_data: pd.DataFrame,
    horizons: list[float] = DEFAULT_TRAIN_HORIZONS_HOURS,
    n_trials: int = N_TRIALS,
) -> tuple[
    OpenstfRegressor, ModelSpecificationDataClass, Report, dict, int, dict[str, Any]
]:
    """Optimize hyperparameters pipeline core.

    Expected prediction job key's: "name", "model"

    Args:
        pj: Prediction job
        input_data: Raw training input data
        horizons: horizons for feature engineering in hours.
        n_trials: The number of trials. Defaults to N_TRIALS.

    Raises:
        ValueError: If the input_date is insufficient.

    Returns:
        - Best model,
        - Model specifications of the best model,
        - Report of the best training round,
        - Trials,
        - Best trial number,
        - Optimized hyperparameters.

    """
    if input_data.empty:
        raise InputDataInsufficientError("Input dataframe is empty")
    elif "load" not in input_data.columns:
        raise InputDataWrongColumnOrderError(
            "Missing the load column in the input dataframe"
        )

    # Validate and clean data
    validated_data = validation.drop_target_na(
        validation.validate(
            pj["id"],
            input_data,
            pj["flatliner_threshold_minutes"],
            pj["resolution_minutes"],
        )
    )

    # Check if sufficient data is left after cleaning
    if not validation.is_data_sufficient(
        validated_data, pj["completeness_threshold"], pj["minimal_table_length"]
    ):
        raise InputDataInsufficientError(
            f"Input data is insufficient for {pj['name']} after validation and cleaning"
        )

    if pj.default_modelspecs:
        feature_names = pj.default_modelspecs.feature_names
        feature_modules = pj.default_modelspecs.feature_modules
    else:
        feature_names = None
        feature_modules = []

    if isinstance(horizons, str):
        if horizons not in set(input_data.columns):
            raise ValueError(
                f"The horizon parameter specifies a column name ({horizons}) missing in"
                " the input data."
            )
        else:
            # sort data to avoid same date repeated multiple time
            input_data = input_data.sort_values(horizons)

    validated_data_with_features = TrainFeatureApplicator(
        horizons=horizons, feature_names=feature_names, feature_modules=feature_modules
    ).add_features(validated_data, pj=pj)

    # Adds additional proloaf features to the input data, historic_load (equal to the load, first column)
    if pj["model"] == "proloaf" and "historic_load" not in list(
        validated_data_with_features.columns
    ):
        validated_data_with_features[
            "historic_load"
        ] = validated_data_with_features.iloc[:, 0]
        # Make sure horizons is last column
        temp_cols = validated_data_with_features.columns.tolist()
        new_cols = temp_cols[:-2] + [temp_cols[-1]] + [temp_cols[-2]]
        validated_data_with_features = validated_data_with_features[new_cols]

    # Create objective (NOTE: this is a callable class)
    objective = ObjectiveCreator.create_objective(model_type=pj["model"])

    study, objective = optuna_optimization(
        pj, objective, validated_data_with_features, n_trials
    )

    best_hyperparams = study.best_params
    # The best_model could be accessed via study.user_attrs["best_model"]

    logger.info(
        f"Finished hyperparameter optimization, error objective {study.best_value} "
        f"and params {best_hyperparams}"
    )

    # Add quantiles to hyperparams so they are stored with the model info
    if pj["quantiles"]:
        best_hyperparams.update(quantiles=pj["quantiles"])

    # model specification
    model_specs = ModelSpecificationDataClass(
        id=pj["id"],
        feature_names=list(validated_data_with_features.columns),
        hyper_params=best_hyperparams,
    )

    # Train a model using the regular train pipeline.
    # The train/validation/test split used in hyperparam optimisation
    # is less suitable for an operational model.
    model, report, model_specs, _ = train_model_pipeline_core(
        pj=pj, input_data=input_data, model_specs=model_specs
    )

    trials = objective.get_trial_track()
    best_trial_number = study.best_trial.number

    return model, model_specs, report, trials, best_trial_number, study.best_params


def optuna_optimization(
    pj: PredictionJobDataClass,
    objective: RegressorObjective,
    validated_data_with_features: pd.DataFrame,
    n_trials: int,
) -> tuple[optuna.study.Study, RegressorObjective]:
    """Perform hyperparameter optimization with optuna.

    Args:
        pj: Prediction job
        objective: Objective function for optuna
        validated_data_with_features: cleaned input dataframe
        n_trials: number of optuna trials

    Returns:
        - Optimization study from optuna
        - The objective object used by optuna

    """
    model = ModelCreator.create_model(pj["model"])
    # Apply set to default hyperparameters if they are specified in the pj
    if pj.default_modelspecs:
        valid_hyper_parameters = {
            key: value
            for key, value in pj.default_modelspecs.hyper_params.items()
            if key in model.get_params().keys()
        }
        model.set_params(**valid_hyper_parameters)

    study = optuna.create_study(
        study_name=pj["model"],
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
        direction="minimize",
    )

    # Start with evaluating the default set of parameters,
    # this way the optimization never get worse than the default values
    study.enqueue_trial(objective.get_default_values())

    if pj.train_split_func is None:
        split_func = split_data_train_validation_test
        split_args = {
            "stratification_min_max": pj["model"] != "proloaf",
            "back_test": True,
        }
    else:
        split_func, split_args = pj.train_split_func.load(
            required_arguments=["data", "test_fraction", "validation_fraction"]
        )

    objective = objective(
        model,
        validated_data_with_features,
        split_func=split_func,
        split_args=split_args,
    )

    # Optuna updates the model by itself
    # and the model is the optimized over this finishes
    study.optimize(
        objective,
        n_trials=n_trials,
        callbacks=[_log_study_progress_and_save_best_model],
        show_progress_bar=False,
        timeout=TIMEOUT,
    )

    return study, objective


def _log_study_progress_and_save_best_model(
    study: optuna.study.Study, trial: optuna.trial.FrozenTrial
) -> None:
    # Collect study and trial data
    # trial_index = study.trials.index(trial)
    # best_trial_index = study.trials.index(study.best_trial)
    value = trial.value
    params = trial.params
    duration = (trial.datetime_complete - trial.datetime_start).total_seconds()
    # Log information about this trial
    logger.debug(
        f"Trial {trial.number} finished with value: {value} and parameters: {params}."
        f"Best trial is {study.best_trial.number}. Iteration took {duration} s"
    )
    # If this trial is the best save the model as a study attribute
    if study.best_trial.number == trial.number:
        study.set_user_attr(key="best_model", value=trial.user_attrs["model"])

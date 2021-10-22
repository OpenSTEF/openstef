# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
from pathlib import Path
from typing import List, Union, Tuple

import optuna
import pandas as pd
import structlog
from openstf_dbc.services.prediction_job import PredictionJobDataClass

from openstf.exceptions import (
    InputDataInsufficientError,
    InputDataWrongColumnOrderError,
)
from openstf.feature_engineering.feature_applicator import TrainFeatureApplicator
from openstf.model.model_creator import ModelCreator
from openstf.model.objective import RegressorObjective
from openstf.model.objective_creator import ObjectiveCreator

# This is required to disable the default optuna logger and pass the logs to our own
# structlog logger
from openstf.model.regressors.regressor import OpenstfRegressor
from openstf.model.serializer import PersistentStorageSerializer
from openstf.validation import validation

optuna.logging.enable_propagation()  # Propagate logs to the root logger.
optuna.logging.disable_default_handler()  # Stop showing logs in sys.stderr.

logger = structlog.get_logger(__name__)

# See https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.Study.html#optuna.study.Study.optimize
N_TRIALS: int = 50  # The number of trials.
TIMEOUT: int = 200  # Stop study after the given number of second(s).
TRAIN_HORIZONS: List[float] = [0.25, 24.0]
TEST_FRACTION: float = 0.1
VALIDATION_FRACTION: float = 0.1


def optimize_hyperparameters_pipeline(
    pj: Union[dict, PredictionJobDataClass],
    input_data: pd.DataFrame,
    trained_models_folder: Union[str, Path],
    horizons: List[float] = TRAIN_HORIZONS,
    n_trials: int = N_TRIALS,
) -> dict:
    """Optimize hyperparameters pipeline.

    Expected prediction job key's: "name", "model"

    Args:
        pj (Union[dict, PredictionJobDataClass]): Prediction job
        input_data (pd.DataFrame): Raw training input data
        trained_models_folder (Path): Path where trained models are stored
        horizons (List[float]): horizons for feature engineering.
        n_trials (int, optional): The number of trials. Defaults to N_TRIALS.

    Raises:
        ValueError: If the input_date is insufficient.

    Returns:
        dict: Optimized hyperparameters.
    """
    if input_data.empty:
        raise InputDataInsufficientError("Input dataframe is empty")
    elif "load" not in input_data.columns:
        raise InputDataWrongColumnOrderError(
            "Missing the load column in the input dataframe"
        )

    # Validate and clean data
    validated_data = validation.clean(validation.validate(pj["id"], input_data))

    # Check if sufficient data is left after cleaning
    if not validation.is_data_sufficient(validated_data):
        raise InputDataInsufficientError(
            f"Input data is insufficient for {pj['name']} "
            f"after validation and cleaning"
        )

    input_data_with_features = TrainFeatureApplicator(horizons=horizons).add_features(
        input_data
    )

    # Create serializer
    serializer = PersistentStorageSerializer(trained_models_folder)
    # Start MLflow
    serializer.setup_mlflow(pj["id"])

    # Create objective (NOTE: this is a callable class)
    objective = ObjectiveCreator.create_objective(model_type=pj["model"])

    model, study, objective = optuna_optimization(
        pj, objective, input_data_with_features, n_trials
    )

    logger.info(
        f"Finished hyperparameter optimization, error objective {study.best_value} "
        f"and params {study.best_params}"
    )

    # Save model
    serializer.save_model(
        model,
        pj=pj,
        # In objective we have the data, thus we create the report there
        report=objective.create_report(model=model),
        phase="Hyperparameter_opt",
        trials=objective.get_trial_track(),
        trial_number=study.best_trial.number,
    )

    return study.best_params


def optuna_optimization(
    pj: Union[PredictionJobDataClass, dict],
    objective: RegressorObjective,
    input_data_with_features: pd.DataFrame,
    n_trials: int,
) -> Tuple[OpenstfRegressor, optuna.study.Study, RegressorObjective]:
    """Perform hyperparameter optimization with optuna

    Args:
        pj: Prediction job
        objective: Objective function for optuna
        input_data_with_features: cleaned input dataframe
        n_trials: number of optuna trials

    Returns:
        model (OpenstfRegressor): Optimized model
        study (optuna.study.Study): Optimization study from optuna
        objective : The objective object used by optuna

    """
    model = ModelCreator.create_model(pj["model"])

    objective = objective(
        model,
        input_data_with_features,
    )

    study = optuna.create_study(
        study_name=pj["model"],
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
        direction="minimize",
    )

    # Optuna updates the model by itself
    # and the model is the optimized over this finishes
    study.optimize(
        objective,
        n_trials=n_trials,
        callbacks=[_log_study_progress],
        show_progress_bar=False,
        timeout=TIMEOUT,
    )

    return model, study, objective


def _log_study_progress(
    study: optuna.study.Study, trial: optuna.trial.FrozenTrial
) -> None:
    # Collect study and trial data
    trial_index = study.trials.index(trial)
    best_trial_index = study.trials.index(study.best_trial)
    value = trial.value
    params = trial.params
    duration = (trial.datetime_complete - trial.datetime_start).total_seconds()
    # Log information about this trial
    logger.debug(
        f"Trial {trial_index} finished with value: {value} and parameters: {params}."
        f"Best trial is {best_trial_index}. Iteration took {duration} s"
    )

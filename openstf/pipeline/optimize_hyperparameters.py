# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
import pandas as pd
import optuna
from typing import List
import structlog
from datetime import datetime

from openstf.model.model_creator import ModelCreator
from openstf.model.objective_creator import ObjectiveCreator
from openstf.feature_engineering.feature_applicator import TrainFeatureApplicator
from openstf.validation import validation
from openstf.model.regressors.regressor_interface import OpenstfRegressorInterface

# This is required to disable the default optuna logger and pass the logs to our own
# structlog logger
optuna.logging.enable_propagation()  # Propagate logs to the root logger.
optuna.logging.disable_default_handler()  # Stop showing logs in sys.stderr.

logger = structlog.get_logger(__name__)

# See https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.Study.html#optuna.study.Study.optimize
N_TRIALS: int = 8  # The number of trials.
TIMEOUT: int = 200  # Stop study after the given number of second(s).
TRAIN_HORIZONS: List[float] = [0.25, 24.0]

# The dictonairy can be interpret the following way
# Tuples; E.g. name:( (min, max), logarithmic); for intigers and floats
#           or name:((option1, option2), logarithmic); picks option 2 for lgb and for all others option 1
# List; E.g. name:[option1, option2, ...]; for categorical
default_paramspace: dict = {
    # General parameters
    "learning_rate": ((0.01, 0.2), True),
    "alpha": ((1e-8, 1.0), True),
    "lambda": ((1e-8, 1.0), True),
    "subsample": ((0.5, 0.99), False),
    "min_child_weight": ((1, 6), False),
    "max_depth": ((3, 10), False),
    "colsample_bytree": ((0.5, 1.0), False),
    "max_delta_step": ((1, 10), False),
}
# Important parameters, model specific
# XGB specific
xgb_paramspace: dict = {
    "gamma": ((1e-8, 1.0), True),
    "booster": ["gbtree", "dart"],
    # , "gblinear" gives warnings because it doesn't use { colsample_bytree, gamma, max_delta_step, max_depth, min_child_weight, subsample }
}

# LGB specific
lgb_paramspace: dict = {
    "num_leaves": ((16, 62), False),
    "boosting_type": ["gbdt", "dart", "rf"],
    "tree_learner": ["serial", "feature", "data", "voting"],
    "n_estimators": ((50, 150), False),
    "min_split_gain": ((1e-8, 1), True),
    "subsample_freq": ((1, 10), False),
}


def get_relevant_model_paramspace(
    model: OpenstfRegressorInterface, paramspace: dict
) -> dict:
    """Return the parameters usefull for the model"""
    # list the possible hyperparameters for the model
    list_default_params = model.get_params()

    # Compare the list to the default parameter space
    keys = [x for x in paramspace.keys() if x in list_default_params.keys()]
    # create a dictonairy with the matching parameters
    model_params = {parameter: paramspace[parameter] for parameter in keys}
    return model_params


def optimize_hyperparameters_pipeline(
    pj: dict,
    input_data: pd.DataFrame,
    horizons: List[float] = TRAIN_HORIZONS,
    n_trials: int = N_TRIALS,
) -> dict:
    """Optimize hyperparameters pipeline.

    Expected prediction job key's: "name", "model"

    Args:
        pj (dict): Prediction job
        input_data (pd.DataFrame): Raw training input data
        horizons (List[float]): horizons for feature engineering.
        n_trials (int, optional): The number of trials. Defaults to N_TRIALS.
        timeout (int, optional): Stop study after the given number of second(s).
            Defaults to TIMEOUT.

    Raises:
        ValueError: If the input_date is insufficient.

    Returns:
        dict: Optimized hyperparameters.
    """
    # Validate and clean data
    validated_data = validation.clean(validation.validate(input_data))

    # Check if sufficient data is left after cleaning
    if not validation.is_data_sufficient(validated_data):
        raise ValueError(
            f"Input data is insufficient for {pj['name']} "
            f"after validation and cleaning"
        )

    input_data_with_features = TrainFeatureApplicator(horizons=horizons).add_features(
        input_data
    )

    # Create objective (NOTE: this is a callable class)
    objective = ObjectiveCreator.create_objective(model_type=pj["model"])

    study = optuna.create_study(
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5), direction="minimize"
    )
    model_type = pj["model"]
    model = ModelCreator.create_model(model_type)

    # Combine the default parameters with all model specific parameters
    paramspace = default_paramspace.copy()
    paramspace.update(**xgb_paramspace, **lgb_paramspace)

    model_params = get_relevant_model_paramspace(model, paramspace)

    if model_type == "lgb":

        model_params["objective"] = objective.eval_metric # The objective of lgb is the eval metric
        pruning_function = optuna.integration.LightGBMPruningCallback
        args_eval = {"metric" : objective.eval_metric, "valid_name" : "valid_1"}
        if objective.eval_metric == "mae":

            args_eval["metric"] = "l1"
    else:
        # for other models use the default objective by not setting it.
        pruning_function = optuna.integration.XGBoostPruningCallback
        args_eval = {"observation_key" : "validation_1-{}".format(objective.eval_metric)}

    start_time = datetime.utcnow()

    objective = objective(model, pruning_function, input_data_with_features, model_params, start_time, **args_eval)

    study.optimize(
        objective,
        n_trials=n_trials,
        callbacks=[_log_study_progress],
        show_progress_bar=False,
    )

    optimized_hyperparams = study.best_params
    optimized_error = study.best_value

    logger.info(
        f"Finished hyperparameter optimization, error objective {optimized_error} "
        f"and params {optimized_hyperparams}"
    )

    return optimized_hyperparams


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

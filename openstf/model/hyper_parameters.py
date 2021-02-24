# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

from datetime import datetime, timedelta

import optuna
from ktpbase.database import DataBase
from ktpbase.log import logging

from openstf.model import metrics
from openstf.model.general import pre_process_data
from openstf.model.train import is_data_sufficient
from openstf.model.trainer.creator import ModelTrainerCreator
from openstf.monitoring.teams import post_teams

# Available trainings period durations for optimization
TRAINING_DURATIONS_DAYS = [30, 60, 90, 120]
MAX_AGE_HYPER_PARAMS_DAYS = 31
SHOW_OPTUNA_PROGRESS_BAR = False


def optimize_hyperparameters_pj(pj):
    """Optimized hyperparameters for a specific prediction job.
    First the age of the last stored hyperparameters is determined. If these are older
     than MAX_AGE_HYPER_PARAMS_DAYS, the hyper parameters will be optimized.

    Args:
        pj: (dict) Prediction job
    """

    db = DataBase()

    # Get data of last stored hyper parameters
    last = db.get_hyper_params_last_optimized(pj)

    # Check if hyper parameters are older than limit
    limit = datetime.utcnow() - timedelta(days=MAX_AGE_HYPER_PARAMS_DAYS)
    if last < limit:
        optimize_hyperparameters(pj["id"])


def optimize_hyperparameters(pid, n_trials=150, datetime_end=datetime.utcnow()):
    """Optimize the hyper parameters for a given prediction job.

        This function optimizes the model specific hyperparameters,
        featureset and training period. After optimization the hyperparameters
        are stored in the database and finally a teams messsage with the results
        is posted.

    Args:
        pid (int): Prediction job ID
        n_trials (int): Maximum number of trials/iteration first stage of
            hyper parameter optimization

    Returns:
        (dict): Dictionary with optimized hyperparameters, keys of the dictionarry
            depend on the model type.
    """

    db = DataBase()
    # initialize logging
    logger = logging.get_logger(__name__)

    max_training_period_days = max(TRAINING_DURATIONS_DAYS)
    # We do not want to train components during hyperparameter optimalisation
    train_components = 0

    # Get prediciton job
    pj = db.get_prediction_job(pid)

    pj["train_components"] = train_components

    # Specify training period
    datetime_start = datetime_end - timedelta(days=max_training_period_days)

    # Get data from database
    data = db.get_model_input(
        pid=pj["id"],
        location=[pj["lat"], pj["lon"]],
        datetime_start=datetime_start,
        datetime_end=datetime_end,
    )
    featuresets = db.get_featuresets()
    # Pre-process data
    clean_data_with_features = pre_process_data(data)

    # Initialize model trainer creator factory object
    mc = ModelTrainerCreator(pj)
    # Initialize model trainer
    model_trainer = mc.create_model_trainer()

    # Check if we have enough data left to continue
    if is_data_sufficient(clean_data_with_features) is False:
        raise ValueError("Input data quality insufficient, aborting!")

    # Setup optuna study
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    # TODO make use of integrated XGBoost callbacks for efficient pruning and
    # faster optimalization runs, when bug in optana is solved
    first_stage_study = optuna.create_study(
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5), direction="minimize"
    )
    second_stage_study = optuna.create_study(
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5), direction="minimize"
    )
    # Carry out hyper-parameters and featureset optimization study
    first_stage_study.optimize(
        lambda trial: model_trainer.hyper_params_objective(
            trial,
            metrics.mae,
            clean_data_with_all_features=clean_data_with_features,
            featuresets=featuresets,
        ),
        n_trials=n_trials,
        show_progress_bar=SHOW_OPTUNA_PROGRESS_BAR,
    )
    optimized_error = first_stage_study.best_trial.value
    logger.info(
        f"Optimized error before training-period optimization: {optimized_error}",
        optimized_error=optimized_error,
    )

    # Carry out training-period optimization study
    second_stage_study.optimize(
        lambda trial: model_trainer.training_period_objective(
            trial,
            metrics.mae,
            unprocessed_data=data,
            training_durations_days=TRAINING_DURATIONS_DAYS,
            # The training-period optimization makes use of the previously optimized parameters
            optimized_parameters=first_stage_study.best_trial.params,
            featuresets=featuresets,
        ),
        n_trials=8,  # Ideally you'd test every duration only once but optuna does not work that way
        timeout=200,
        show_progress_bar=SHOW_OPTUNA_PROGRESS_BAR,
    )
    optimized_parameters = {
        **first_stage_study.best_trial.params,
        **second_stage_study.best_trial.params,
    }
    optimized_error = second_stage_study.best_trial.value
    logger.info("Optimized parameters", optimized_parameters=optimized_parameters)
    logger.info("Final optimized error", optimized_error=optimized_error)

    # Write best hyperparameters to database
    db.write_hyper_params(pj, optimized_parameters)
    # Compose message for teams (this is still cgb specific)
    msg = format_teams_message(pj, optimized_parameters)
    # Sent message to Teams
    post_teams(msg)

    return optimized_parameters


# NOTE perhaps formatting a message with paramters
# does not belong to the responsibility of this module
def format_teams_message(pj, hyper_params):
    color = "#046b00"  # green

    # format all hyper parameters using limited precision for floats
    # make keys bold (**key**)
    values = []
    for k, v in hyper_params.items():
        if type(v) is float:
            values.append(f"**{k}**: {v:0.3f}")
            continue
        values.append(f"**{k}**: {v}")
    # join all {hyper_parameter_name}: {value}  pairs with a new line
    text = "".join([f"* {v}   \n" for v in values])

    msg = {
        "fallback": "Optimized hyperparameters for model {0} {1}".format(
            pj["name"], pj["description"]
        ),
        "title": "Optimized hyperparameters for {0} {1}".format(
            pj["name"], pj["description"]
        ),
        "text": text,
        "color": color,
    }
    return msg

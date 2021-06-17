from openstf.model.objective_creator import ObjectiveCreator
import pandas as pd
import optuna
import structlog

from openstf.feature_engineering.feature_applicator import TrainFeatureApplicator
from openstf.validation import validation

logger = structlog.get_logger(__name__)

# See https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.Study.html#optuna.study.Study.optimize
N_TRIALS: int = 8   # The number of trials.
TIMEOUT: int = 200  # Stop study after the given number of second(s).

def optimize_hyperparameters_pipeline(
    pj: dict,
    input_data: pd.DataFrame,
    n_trials: int = N_TRIALS,
    timeout: int = TIMEOUT
):
    """Optimize hyperparameters pipeline.

    Args:
        pj (dict): Prediction job
        input_data (pd.DataFrame): Raw training input data
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

    input_data_with_features = TrainFeatureApplicator(horizons=[0.25, 24]).add_features(
        input_data
    )

    # Create relevant model
    Objective = ObjectiveCreator.create_objective(model_type=pj["model"])

    study = optuna.create_study(
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5), direction="minimize"
    )

    study.optimize(
        Objective(input_data_with_features),
        n_trials=n_trials,
        timeout=timeout,
        callbacks=[_log_study_progress],
        show_progress_bar=False
    )

    optimized_hyperparams = study.best_params
    optimized_error = study.best_value

    logger.info(
        f"Finished hyperparameter optimization, error objective {optimized_error} "
        f"and params {optimized_hyperparams}"
    )

    return optimized_hyperparams

def _log_study_progress(study, trial):
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
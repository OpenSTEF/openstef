# SPDX-FileCopyrightText: 2017-2023 Contributors to the OpenSTEF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

"""This module contains the CRON job that is periodically executed to calculate key performance indicators (KPIs).

This code assumes prognoses are available from the persistent storage.
If these are not available run create_forecast.py to train all models.

The folowing tasks are caried out:
  1: Calculate the KPI for a given pid. Ignore SplitEnergy
  2: Create figures
  3: Write KPI to database

Example:
    This module is meant to be called directly from a CRON job.
    Alternatively this code can be run directly by running::
        $ python calculate_kpi.py

"""
import logging

# Import builtins
from datetime import datetime, timedelta, UTC
from pathlib import Path

import numpy as np
import pandas as pd
import structlog

from openstef.data_classes.prediction_job import PredictionJobDataClass
from openstef.enums import ModelType
from openstef.exceptions import NoPredictedLoadError, NoRealisedLoadError
from openstef.metrics import metrics
from openstef.settings import Settings
from openstef.tasks.utils.predictionjobloop import PredictionJobLoop
from openstef.tasks.utils.taskcontext import TaskContext
from openstef.validation import validation

# Thresholds for retraining and optimizing
THRESHOLD_RETRAINING = 0.25
THRESHOLD_OPTIMIZING = 0.50


def main(model_type: ModelType = None, config=None, database=None) -> None:
    taskname = Path(__file__).name.replace(".py", "")

    if database is None or config is None:
        raise RuntimeError(
            "Please specifiy a config object and/or database connection object. These"
            " can be found in the openstef-dbc package."
        )

    if model_type is None:
        model_type = [ml.value for ml in ModelType]

    with TaskContext(taskname, config, database) as context:
        # Set start and end time
        end_time = datetime.now(tz=UTC)
        start_time = end_time - timedelta(days=1)

        PredictionJobLoop(context, model_type=model_type).map(
            check_kpi_task,
            context,
            start_time=start_time,
            end_time=end_time,
        )


def check_kpi_task(
    pj: PredictionJobDataClass,
    context: TaskContext,
    start_time: datetime,
    end_time: datetime,
    threshold_optimizing=THRESHOLD_OPTIMIZING,
    threshold_retraining=THRESHOLD_RETRAINING,
) -> None:
    # Apply default parameters if none are provided
    if start_time is None:
        start_time = datetime.now(tz=UTC) - timedelta(days=1)
    if end_time is None:
        end_time = datetime.now(tz=UTC)

    # Get realised load data
    realised = context.database.get_load_pid(pj["id"], start_time, end_time, "15T")

    # Get predicted load
    predicted_load = context.database.get_predicted_load_tahead(
        pj, start_time, end_time
    )

    # Get basecase prediction
    load_1_week_before = context.database.get_load_pid(
        pj["id"], start_time - timedelta(days=7), end_time - timedelta(days=7), "15T"
    )
    if len(load_1_week_before) > 0:
        basecase = load_1_week_before.shift(periods=7, freq="d")
    else:
        basecase = pd.DataFrame()

    kpis = calc_kpi_for_specific_pid(pj["id"], realised, predicted_load, basecase)
    # Write KPI's to database
    context.database.write_kpi(pj, kpis)

    # Add pid to the list of pids that should be retrained or optimized if
    # performance is insufficient
    if kpis["47.0h"]["rMAE"] > threshold_retraining:
        context.logger.warning(
            "Need to retrain model, retraining threshold rMAE 47h exceeded",
            t_ahead="47.0h",
            rMAE=kpis["47.0h"]["rMAE"],
            retraining_threshold=threshold_retraining,
        )

    if kpis["47.0h"]["rMAE"] > threshold_optimizing:
        context.logger.warning(
            "Need to optimize hyperparameters, optimizing threshold rMAE 47h exceeded",
            t_ahead="47.0h",
            rMAE=kpis["47.0h"]["rMAE"],
            optimizing_threshold=threshold_optimizing,
        )


def calc_kpi_for_specific_pid(
    pid: int,
    realised: pd.DataFrame,
    predicted_load: pd.DataFrame,
    basecase: pd.DataFrame,
) -> dict:
    """Function that checks the model performance based on a pid. This function.

    - loads and combines forecast and realised data
    - calculated several key performance indicators (KPIs)
    These metric include:
        - RMSE,
        - bias,
        - NSME (model efficiency, between -inf and 1)
        - Mean absolute Error

    Args:
        pid: Prediction ID for a given prediction job
        realised: Realised load.
        predicted_load: Predicted load.
        basecase: Basecase predicted load.

    Returns:
        - Dictionary that includes a dictonary for each t_ahead.
        - Dict includes enddate en window (in days) for clarification

    Raises:
        NoPredictedLoadError: When no predicted load for given datatime range.
        NoRealisedLoadError: When no realised load for given datetime range.

    Example:
        To get the rMAE for the 24 hours ahead prediction: kpis['24h']['rMAE']

    """
    COMPLETENESS_REALISED_THRESHOLDS = 0.7
    COMPLETENESS_PREDICTED_LOAD_THRESHOLD = 0.7

    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelName(Settings.log_level)
        )
    )
    logger = structlog.get_logger(__name__)

    # If predicted is empty
    if len(predicted_load) == 0:
        raise NoPredictedLoadError(pid)

    # If realised is empty
    if len(realised) == 0:
        raise NoRealisedLoadError(pid)

    # Define start and end time
    start_time = realised.index.min().to_pydatetime()
    end_time = realised.index.max().to_pydatetime()

    completeness_realised = validation.calc_completeness_dataframe(realised)[0]

    # Interpolate missing data if needed
    realised = realised.resample("15T").interpolate(limit=3)

    completeness_predicted_load = validation.calc_completeness_dataframe(predicted_load)

    # Combine the forecast and the realised to make sure indices are matched nicely
    combined = pd.merge(realised, predicted_load, left_index=True, right_index=True)

    # Add basecase (load in same time period 7 days ago)
    # Check if basecase is not empty, else make a dummy dataframe
    if len(basecase) == 0:
        basecase = pd.DataFrame(columns=["load"])
    basecase = basecase.rename(columns=dict(load="basecase"))

    combined = combined.merge(basecase, how="left", left_index=True, right_index=True)

    # Raise exception in case of constant load
    if combined.load.nunique() == 1:
        logger.warning(
            "The load is constant! KPIs will still be calculated, but relative metrics"
            " will be nan."
        )

    # Define output dictonary
    kpis = dict()

    # Extract t_aheads from predicted_load,
    # Make a list of tuples with [(forecast_xh, stdev_xh),(..,..),..]
    hor_list = [
        ("forecast_" + t_ahead, "stdev_" + t_ahead)
        for t_ahead in set(col.split("_")[1] for col in predicted_load.columns)
    ]

    # cast date to int
    date = pd.to_datetime(end_time)

    # Calculate model metrics and add them to the output dictionary
    logger.info("Start calculating kpis")
    for hor_cols in hor_list:
        t_ahead_h = hor_cols[0].split("_")[1]
        fc = combined[hor_cols[0]]  # load predictions
        st = combined[hor_cols[1]]  # standard deviations of load predictions

        completeness_predicted_load_specific_hor = (
            validation.calc_completeness_dataframe(fc.to_frame(name=t_ahead_h))[0]
        )
        kpis.update(
            {
                t_ahead_h: {
                    "RMSE": metrics.rmse(combined["load"], fc),
                    "bias": metrics.bias(combined["load"], fc),
                    "NSME": metrics.nsme(combined["load"], fc),
                    "MAE": metrics.mae(combined["load"], fc),
                    "rMAE": metrics.r_mae(combined["load"], fc),
                    "rMAE_highest": metrics.r_mae_highest(combined["load"], fc),
                    "rMNE_highest": metrics.r_mne_highest(combined["load"], fc),
                    "rMPE_highest": metrics.r_mpe_highest(combined["load"], fc),
                    "rMAE_lowest": metrics.r_mae_lowest(combined["load"], fc),
                    "skill_score_basecase": metrics.skill_score(
                        combined["load"],
                        combined["basecase"],
                        np.mean(combined["basecase"]),
                    ),
                    "skill_score": metrics.skill_score(
                        combined["load"], fc, np.mean(combined["basecase"])
                    ),
                    "skill_score_positive_peaks": metrics.skill_score_positive_peaks(
                        combined["load"], fc, np.mean(combined["basecase"])
                    ),
                    "skill_score_positive_peaks_basecase": metrics.skill_score_positive_peaks(
                        combined["load"],
                        combined["basecase"],
                        np.mean(combined["basecase"]),
                    ),
                    "franks_skill_score": metrics.franks_skill_score(
                        combined["load"], fc, combined["basecase"]
                    ),
                    "franks_skill_score_peaks": metrics.franks_skill_score_peaks(
                        combined["load"], fc, combined["basecase"]
                    ),
                    "load_range": combined["load"].max() - combined["load"].min(),
                    "frac_in_1sdev": metrics.frac_in_stdev(combined["load"], fc, st),
                    "frac_in_2sdev": metrics.frac_in_stdev(
                        combined["load"], fc, 2 * st
                    ),
                    "completeness_realised": completeness_realised,
                    "completeness_predicted": completeness_predicted_load_specific_hor,
                    "date": date,  # cast to date
                    "window_days": np.round(
                        (end_time - start_time).total_seconds() / 60.0 / 60.0 / 24.0
                    ),
                }
            }
        )

        if completeness_realised < COMPLETENESS_REALISED_THRESHOLDS:
            logger.warning(
                "Completeness realised load too low",
                prediction_id=pid,
                start_time=start_time,
                end_time=end_time,
                completeness=completeness_realised,
                completeness_threshold=COMPLETENESS_REALISED_THRESHOLDS,
            )
            set_incomplete_kpi_to_nan(kpis, t_ahead_h)
        if completeness_predicted_load.any() < COMPLETENESS_PREDICTED_LOAD_THRESHOLD:
            logger.warning(
                "Completeness predicted load of specific horizon too low",
                prediction_id=pid,
                horizon=t_ahead_h,
                start_time=start_time,
                end_time=end_time,
                completeness=completeness_predicted_load,
                completeness_threshold=COMPLETENESS_PREDICTED_LOAD_THRESHOLD,
            )
            set_incomplete_kpi_to_nan(kpis, t_ahead_h)

    # Return output dictionary
    return kpis


def set_incomplete_kpi_to_nan(kpis: dict, t_ahead_h: str) -> None:
    """Checks the given kpis for completeness and sets to nan if this not true.

    Args:
        kpis: the kpis
        t_ahead_h: t_ahead_h

    """
    kpi_metrics = list(kpis[t_ahead_h].keys())
    # Set to nan
    for kpi in kpi_metrics:
        if kpi not in [
            "completeness_realised",
            "completeness_predicted",
            "date",
            "window_days",
        ]:
            kpis[t_ahead_h].update({kpi: np.nan})


if __name__ == "__main__":
    main()

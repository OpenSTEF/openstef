# SPDX-FileCopyrightText: 2017-2023 Contributors to the OpenSTEF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
"""This module contains the CRON job that is periodically executed to make prognoses and save them in to the database.

This code assumes trained models are available from the persistent storage. If these
are not available run model_train.py to train all models.
To provide the prognoses the folowing steps are carried out:
  1. Get historic training data (TDCV, Load, Weather and day_ahead_electricity_price price data)
  2. Apply features
  3. Load model
  4. Make prediction
  5. Write prediction to the database
  6. Send Teams message if something goes wrong

Example:
    This module is meant to be called directly from a CRON job.
    Alternatively this code can be run directly by running::

        $ python create_forecast.py

"""
from datetime import datetime, timedelta, UTC
from pathlib import Path

from openstef.data_classes.prediction_job import PredictionJobDataClass
from openstef.enums import BiddingZone, ModelType, PipelineType
from openstef.exceptions import InputDataOngoingZeroFlatlinerError
from openstef.pipeline.create_forecast import create_forecast_pipeline
from openstef.tasks.utils.predictionjobloop import PredictionJobLoop
from openstef.tasks.utils.taskcontext import TaskContext
from openstef.validation.validation import detect_ongoing_zero_flatliner

T_BEHIND_DAYS: int = 14


def create_forecast_task(
    pj: PredictionJobDataClass, context: TaskContext, t_behind_days: int = T_BEHIND_DAYS
) -> None:
    """Top level task that creates a forecast.

    On this task level all database and context manager dependencies are resolved.

    Expected prediction job keys; "id", "lat", "lon", "resolution_minutes",
        "horizon_minutes", "type", "name", "quantiles"

    Args:
        pj: Prediction job
        context: Contect object that holds a config manager and a database connection
        t_behind_days: number of days included as history. This is used to generated lagged features for the to-be-forecasted period

    """
    # Check pipeline types
    if PipelineType.FORECAST not in pj.pipelines_to_run:
        context.logger.info(
            "Skip this PredictionJob because forecast pipeline is not specified in the pj."
        )
        return

    # TODO: Improve implementation by using a field in the database and leveraging the
    #       `pipelines_to_run` attribute of the `PredictionJobDataClass` object. This
    #       would require a change to the MySQL datamodel.
    if (
        context.config.externally_posted_forecasts_pids
        and pj.id in context.config.externally_posted_forecasts_pids
    ):
        context.logger.info(
            "Skip this PredictionJob because its forecasts are posted by an external process."
        )
        return

    # Extract mlflow tracking URI and trained models folder
    mlflow_tracking_uri = context.config.paths_mlflow_tracking_uri

    # Define datetime range for input data
    datetime_start = datetime.now(tz=UTC) - timedelta(days=t_behind_days)
    datetime_end = datetime.now(tz=UTC) + timedelta(seconds=pj.horizon_minutes * 60)

    # Retrieve input data
    input_data = context.database.get_model_input(
        pid=pj["id"],
        location=[pj["lat"], pj["lon"]],
        datetime_start=datetime_start,
        datetime_end=datetime_end,
        market_price=pj.electricity_bidding_zone.value,
    )

    # Add APX price to the input data for backward compatibility,remove this line when all models are retrained
    if pj.electricity_bidding_zone == BiddingZone.NL:
        input_data["APX"] = input_data["day_ahead_electricity_price"]

    try:
        # Make forecast with the forecast pipeline
        forecast = create_forecast_pipeline(
            pj, input_data, mlflow_tracking_uri=mlflow_tracking_uri
        )
    except (InputDataOngoingZeroFlatlinerError, LookupError) as e:
        if (
            context.config.known_zero_flatliners
            and pj.id in context.config.known_zero_flatliners
        ):
            context.logger.info(
                "No forecasts were made for this known zero flatliner prediction job. No forecasts need to be made either, since the fallback forecasts are sufficient."
            )
            return
        elif isinstance(e, InputDataOngoingZeroFlatlinerError):
            raise InputDataOngoingZeroFlatlinerError(
                'All recent load measurements are zero. Check the load profile of this pid as well as related/neighbouring prediction jobs. Afterwards, consider adding this pid to the "known_zero_flatliners" app_setting and possibly removing other pids from the same app_setting.'
            ) from e
        elif isinstance(e, LookupError):
            zero_flatliner_ongoing = detect_ongoing_zero_flatliner(
                load=input_data.iloc[:, 0],
                duration_threshold_minutes=pj.flatliner_threshold_minutes,
            )
            if zero_flatliner_ongoing:
                raise LookupError(
                    'Model not found. Consider checking for a zero flatliner and adding this pid to the "known_zero_flatliners" app_setting. For zero flatliners, no model can be trained.'
                ) from e
            else:
                raise e

    # Write forecast to the database
    context.database.write_forecast(forecast, t_ahead_series=True)


def main(model_type=None, config=None, database=None, **kwargs):
    taskname = Path(__file__).name.replace(".py", "")

    if database is None or config is None:
        raise RuntimeError(
            "Please specify a config object and/or database connection object. These"
            " can be found in the openstef-dbc package."
        )

    with TaskContext(taskname, config, database) as context:
        if model_type is None:
            model_type = [ml.value for ml in ModelType]

        PredictionJobLoop(context, model_type=model_type).map(
            create_forecast_task, context, **kwargs
        )


if __name__ == "__main__":
    main()

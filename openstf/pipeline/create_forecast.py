# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

from datetime import datetime, timedelta, timezone

import pandas as pd
from ktpbase.database import DataBase
import structlog

from openstf.validation import validation
from openstf.preprocessing import preprocessing
from openstf.postprocessing import postprocessing
from openstf.feature_engineering.feature_applicator import (
    OperationalPredictFeatureApplicator,
)
from openstf.enums import ForecastType
from openstf.model.prediction.creator import PredictionModelCreator

# configuration constants
FLATLINER_THRESHOLD = 6
COMPLETENESS_THRESHOLD = 0.7
FEATURES_H_AHEAD = 0.25
# cache input (for components predictions)
_input_data_cache = {}


def create_forecast_pipeline(pj, forecast_type=ForecastType.DEMAND):
    logger = structlog.get_logger(__name__)
    logger.info(
        "Start making prediction",
        prediction_id=pj["id"],
        customer_name=pj["name"],
        prediction_type="prediction",
    )
    # preparation ######################################################################
    forecast_start, forecast_end = generate_forecast_datetime_range(
        resolution_minutes=pj["resolution_minutes"],
        horizon_minutes=pj["horizon_minutes"],
    )
    datetime_start, datetime_end = generate_inputdata_datetime_range(
        t_behind_days=14, t_ahead_days=3
    )

    prediction_model = PredictionModelCreator.create_prediction_model(
        pj=pj, forecast_type=forecast_type
    )

    # load input data from database ###################################################
    input_data = get_model_input(
        pj=pj, datetime_start=datetime_start, datetime_end=datetime_end
    )
    # pre process input data ##########################################################
    preprocessed_input_data = pre_process_input_data(input_data, FLATLINER_THRESHOLD)

    # feature engineering #############################################################
    input_data_with_features = OperationalPredictFeatureApplicator(
        features=prediction_model.feature_names, horizons=[FEATURES_H_AHEAD]
    ).add_features(preprocessed_input_data)
    # make forecast ###################################################################
    # Create correct format for to-be-forecasted times
    forecast_input_data = input_data_with_features.loc[
        forecast_start:forecast_end, prediction_model.feature_names
    ]
    completeness = prediction_model.calculate_completeness(forecast_input_data)
    # make forecast or fallback forecast
    if is_complete_enough(completeness, COMPLETENESS_THRESHOLD) is True:
        forecast = prediction_model.make_forecast(forecast_input_data)
    else:
        forecast = prediction_model.make_fallback_forecast(
            forecast_input_data=forecast_input_data, load_data=input_data[["load"]]
        )

    # post process forecast ###########################################################
    # Add wind and solar components
    forecast["forecast"] = postprocessing.post_process_wind_solar(
        forecast["forecast"], forecast_type
    )

    # save forecast to database #######################################################
    DataBase().write_forecast(forecast, t_ahead_series=True)

    return forecast


def make_components_prediction(pj):
    logger = structlog.get_logger(__name__)
    logger.info("Make components prediction", prediction_id=pj["id"])
    # select time period for the coming two days
    datetime_start, datetime_end = generate_inputdata_datetime_range(
        t_behind_days=0, t_ahead_days=3
    )
    logger.info(
        "Get predicted load", datetime_start=datetime_start, datetime_end=datetime_end
    )
    # Get most recent load forecast
    forecast = DataBase().get_predicted_load(
        pj, start_time=datetime_start, end_time=datetime_end
    )
    # Check if forecast is not empty
    if len(forecast) == 0:
        logger.warning(f'No forecast found. Skipping pid {pj["id"]}')
        return

    forecast["pid"] = pj["id"]
    forecast["customer"] = pj["name"]
    forecast["description"] = pj["description"]
    forecast["type"] = pj["typ"]
    forecast = forecast.drop(["stdev"], axis=1)

    logger.info("retrieving weather data")
    # Get required weather data
    weather_data = DataBase().get_weather_data(
        [pj["lat"], pj["lon"]],
        ["radiation", "windspeed_100m"],
        datetime_start=datetime_start,
        datetime_end=datetime_end,
        source="optimum",
    )

    # Get splitting coeficients
    split_coefs = DataBase().get_energy_split_coefs(pj)

    if len(split_coefs) == 0:
        logger.warning(f'No Coefs found. Skipping pid {forecast["pid"]}')
        return

    # Make component forecasts
    try:
        forecasts = postprocessing.split_forecast_in_components(
            forecast, weather_data, split_coefs
        )
    except Exception as e:
        # In case something goes wrong we fall back on aan empty dataframe
        logger.warning(
            f"Could not make component forecasts: {e}, falling back on series of zeros!",
            exc_info=e,
        )
        forecasts = pd.DataFrame()

    # save forecast to database #######################################################
    DataBase().write_forecast(forecasts)
    logger.debug("Written forecast to database")

    return forecasts


def make_basecase_prediction(pj):
    logger = structlog.get_logger(__name__)
    fill_limit = 4
    # preparation ######################################################################
    datetime_start, datetime_end = generate_inputdata_datetime_range(
        t_behind_days=15, t_ahead_days=0
    )
    logger.debug("Generated input timestamps")

    prediction_model = PredictionModelCreator.create_prediction_model(
        pj=pj, forecast_type=ForecastType.BASECASE
    )
    logger.debug("Loaded prediction model")
    forecast_resolution = f'{pj["resolution_minutes"]}T'

    # load input data from database ###################################################
    # Get historic load
    # TODO what to do when no load data is available
    historic_load = DataBase().get_load_pid(
        pid=pj["id"],
        datetime_start=datetime_start,
        datetime_end=datetime_end,
        forecast_resolution=forecast_resolution,
    )
    logger.debug("Retrieved historic load")

    # pre process input data ##########################################################

    # resample, fill missing values up to fill_limit
    historic_load = (
        historic_load.resample(forecast_resolution).mean().interpolate(limit=fill_limit)
    )
    # make forecast ###################################################################
    forecast = prediction_model.make_basecase_forecast(pj, historic_load)
    logger.debug("Made actual basecase prediction")
    # post process forecast ###########################################################

    # save forecast to database #######################################################
    DataBase().write_forecast(forecast, t_ahead_series=True)
    logger.debug("Written forecast to database")

    return forecast


# Helpers #############################################################################
# preparation
def generate_inputdata_datetime_range(t_behind_days=14, t_ahead_days=3):
    # get current date UTC
    date_today_utc = datetime.now(timezone.utc).date()
    # Date range for input data
    datetime_start = date_today_utc - timedelta(days=t_behind_days)
    datetime_end = date_today_utc + timedelta(days=t_ahead_days)

    return datetime_start, datetime_end


def generate_forecast_datetime_range(resolution_minutes, horizon_minutes):
    # get current date and time UTC
    datetime_utc = datetime.now(timezone.utc)
    # Datetime range for time interval to be predicted
    forecast_start = datetime_utc - timedelta(minutes=resolution_minutes)
    forecast_end = datetime_utc + timedelta(minutes=horizon_minutes)

    return forecast_start, forecast_end


# get data
def get_model_input(pj, datetime_start, datetime_end):
    # generate an unique identifier for storing/loading model input data
    # by doing so we don't have to retrieve the input data from the database
    # when making component predictions
    identifier = generate_input_data_id(pj, datetime_start, datetime_end)
    if identifier not in _input_data_cache.keys():
        input_data = DataBase().get_model_input(
            pid=pj["id"],
            location=(pj["lat"], pj["lon"]),
            # TODO it makes more sense to do the string converion in the
            # DataBase class and use pure Python datatypes in the rest of the code
            datetime_start=str(datetime_start),
            datetime_end=str(datetime_end),
        )
        # cache input data
        # TODO perhaps we can come up with something better? Or leave this to the
        # database. Perhaps its cheap anyway?
        _input_data_cache[identifier] = input_data
    else:
        input_data = _input_data_cache[identifier]

    return input_data


def generate_input_data_id(pj, datetime_start, datetime_end):
    return f'{pj["id"]}_{pj["lat"]}_{pj["lon"]}_{datetime_start}_{datetime_end}'


def _clear_input_data_cache():
    """Clear the input data cache dictionairy.

    This is mainly useful for testing.
    """
    global _input_data_cache
    _input_data_cache = {}


# pre processing
def pre_process_input_data(input_data, flatliner_threshold):
    logger = structlog.get_logger(__name__)
    # Check for repeated load observations due to invalid measurements
    suspicious_moments = validation.find_nonzero_flatliner(
        input_data, threshold=flatliner_threshold
    )
    if suspicious_moments is not None:
        # Covert repeated load observations to NaN values
        input_data = preprocessing.replace_invalid_data(input_data, suspicious_moments)
        # Calculate number of NaN values
        # TODO should this not be part of the replace_invalid_data function?
        num_nan = sum([True for i, row in input_data.iterrows() if all(row.isnull())])
        logger.warning(
            "Found suspicious data points, converted to NaN value",
            num_nan_values=num_nan,
        )

    return input_data


# other
def is_complete_enough(completeness, completeness_threshold):
    logger = structlog.get_logger(__name__)

    if completeness < completeness_threshold:
        logger.warning(
            "Forecast data completeness too low",
            completeness=completeness,
            completeness_threshold=completeness_threshold,
        )
        return False

    return True

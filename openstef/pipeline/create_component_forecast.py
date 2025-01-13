# SPDX-FileCopyrightText: 2017-2023 Contributors to the OpenSTEF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

import logging

import joblib
import numpy as np
import pandas as pd
import structlog

import openstef.postprocessing.postprocessing as postprocessing
from openstef import PROJECT_ROOT
from openstef.data_classes.prediction_job import PredictionJobDataClass
from openstef.enums import ForecastType
from openstef.model.regressors.dazls import Dazls
from openstef.settings import Settings

# Set the path for the Dazls stored model
DAZLS_STORED = str(
    PROJECT_ROOT / "openstef" / "data" / "dazls_model_3.4.24" / "dazls_stored_3.4.24_"
)


def create_input(
    pj: PredictionJobDataClass, input_data: pd.DataFrame, weather_data: pd.DataFrame
) -> pd.DataFrame:
    """This function prepares the input data.

    This data will be used for the Dazls model prediction, so they will be
    according Dazls model requirements.

    Args:
        pj: Prediction job
        input_data: Input forecast for the components forecast.
        weather_data: Weather data with 'radiation' and 'windspeed_100m' columns

    Returns:
        It outputs a dataframe which will be used for the Dazls prediction function.

    """
    # Prepare raw input data
    input_df = (
        weather_data[["radiation", "windspeed_100m"]]
        .merge(
            input_data[["forecast"]].rename(columns={"forecast": "total_load"}),
            how="inner",
            right_index=True,
            left_index=True,
        )
        .dropna()
    )
    # Add additional features
    input_df["lat"] = pj["lat"]
    input_df["lon"] = pj["lon"]

    input_df["solar_on"] = 1
    input_df["wind_on"] = 1
    input_df["hour"] = input_df.index.hour
    input_df["minute"] = input_df.index.minute

    input_df["var0"] = input_df["total_load"].var()
    input_df["var1"] = input_df["radiation"].var()
    input_df["var2"] = input_df["windspeed_100m"].var()

    input_df["sem0"] = input_df["total_load"].sem()
    input_df["sem1"] = input_df["radiation"].sem()
    input_df["sem2"] = input_df["windspeed_100m"].sem()

    # Features for the new model
    # Periodic Month feature
    c = (1 / 11) * np.pi - (1 / 365)
    n = np.array(input_df.index.month, dtype=float)
    input_df["month_ff"] = np.sin(c * (n - 1))

    return input_df


def create_components_forecast_pipeline(
    pj: PredictionJobDataClass, input_data: pd.DataFrame, weather_data: pd.DataFrame
) -> pd.DataFrame:
    """Pipeline for creating a component forecast using Dazls prediction model.

    Args:
        pj: Prediction job
        input_data: Input forecast for the components forecast.
        weather_data: Weather data with 'radiation' and 'windspeed_100m' columns

    Returns:
        DataFrame with component forecasts. The dataframe contains these columns;
        "forecast_wind_on_shore",
        "forecast_solar",
        "forecast_other",
        "pid",
        "customer",
        "description",
        "type",
        "algtype"

    """
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelName(Settings.log_level)
        )
    )
    logger = structlog.get_logger(__name__)
    logger.info("Make components prediction", pid=pj["id"])

    # Make component forecasts
    try:
        dazls_input_data = create_input(pj, input_data, weather_data)

        # Save and load the model as .sav file (or as .z file)
        # For the code contact: korte.termijn.prognoses@alliander.com
        dazls_model = Dazls()
        dazls_model.model_ = joblib.load(DAZLS_STORED + "baseline_model.z")

        logger.info("DAZLS model loaded", dazls_model=str(dazls_model))

        # Use the predict function of Dazls model
        # As input data we use the input_data function which takes into consideration what we want as an input for the forecast and what Dazls can accept as an input
        forecasts = dazls_model.predict(x=dazls_input_data)

        # Set the columns for the output forecast dataframe
        forecasts = pd.DataFrame(
            forecasts,
            columns=["forecast_wind_on_shore", "forecast_solar"],
            index=dazls_input_data.index,
        )

        # Make post-processed forecasts for solar and wind power
        # These forecasts are respectively for the components: "forecast_solar" and "forecast_wind_on_shore"
        # The outcome forecasts are added in the "forecasts" DataFrame we created above
        forecasts["forecast_solar"] = postprocessing.post_process_wind_solar(
            forecasts["forecast_solar"], forecast_type=ForecastType.SOLAR
        )
        forecasts["forecast_wind_on_shore"] = postprocessing.post_process_wind_solar(
            forecasts["forecast_wind_on_shore"], forecast_type=ForecastType.WIND
        )

        # Make forecast for the component: "forecast_other"
        forecasts["forecast_other"] = (
            dazls_input_data["total_load"]
            - forecasts["forecast_solar"]
            - forecasts["forecast_wind_on_shore"]
        )

        # Make sure the forecasts have the same form as the input data. Pad with 0 if necessary
        forecasts = forecasts.reindex(index=input_data.index, fill_value=0)
    except Exception as e:
        # In case something goes wrong we fall back on an a zero-filled dataframe
        logger.warning(
            f"Could not make component forecasts: {e}, falling back on series of"
            " zeros!",
            exc_info=e,
        )
        forecasts = pd.DataFrame(
            data=0,
            index=input_data.index,
            columns=["forecast_wind_on_shore", "forecast_solar", "forecast_other"],
        )

    # Prepare for output
    # Add more prediction properties to the forecast ("pid","customer","description","type","algtype)
    forecasts = postprocessing.add_prediction_job_properties_to_forecast(
        pj, forecasts, algorithm_type="component"
    )
    return forecasts

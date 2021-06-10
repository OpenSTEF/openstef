import structlog

import pandas as pd

import openstf.postprocessing.postprocessing as postprocessing


def create_components_forecast_pipeline(pj, input_data, weather_data, split_coefs):
    """ Pipeline for creating a component forecast

    Args:
        pj (dict): Prediction job
        input_data (pd.DataFrame): Input data fore the components forecast, this usually is a regular forecast
        weather_data (pd.DataFrame): Weather data with 'radiation' and 'windspeed_100m' columns
        split_coefs (dict): coefieicnts for the splitting that are determined earlier

    Returns:
        pd.DataFrame with component forecasts

    """
    logger = structlog.get_logger(__name__)
    logger.info("Make components prediction", prediction_id=pj["id"])

    # Make component forecasts
    try:
        forecasts = postprocessing.split_forecast_in_components(
            input_data, weather_data, split_coefs
        )
    except Exception as e:
        # In case something goes wrong we fall back on aan empty dataframe
        logger.warning(
            f"Could not make component forecasts: {e}, falling back on series of zeros!",
            exc_info=e,
        )
        forecasts = pd.DataFrame()
    else:

        forecasts = forecasts.drop(
            ["stdev"], axis=1
        )  # for component forecasts we do not have a stdev

    # Prepare for output
    forecasts = postprocessing.add_prediction_job_properties_to_forecast(
        pj,
        forecasts,
    )

    return forecasts

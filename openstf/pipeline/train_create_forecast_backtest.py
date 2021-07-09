# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
from typing import List, Tuple


import pandas as pd
from sklearn.base import RegressorMixin

from openstf.model.confidence_interval_applicator import ConfidenceIntervalApplicator
from openstf.postprocessing.postprocessing import (
    add_prediction_job_properties_to_forecast,
)
from openstf.pipeline.train_model import train_pipeline_common

DEFAULT_TRAIN_HORIZONS: List[float] = [0.25, 24.0]
DEFAULT_EARLY_STOPPING_ROUNDS: int = 10


def train_model_and_forecast_back_test(
    pj: dict,
    input_data: pd.DataFrame,
    training_horizons: List[float] = None,
) -> Tuple[pd.DataFrame, RegressorMixin]:
    """Pipeline for a back test.

        DO NOT USE THIS PIPELINE FOR OPERATIONAL FORECASTS

    Args:
        pj (dict): Prediction job.
        input_data (pd.DataFrame): Input data
        training_horizons (list): horizons to train on in hours.
            These horizons are also used to make predictions (one for every horizon)

    Returns:
        forecast (pandas.DataFrame)

    """
    if training_horizons is None:
        training_horizons = DEFAULT_TRAIN_HORIZONS

    # Call common training pipeline
    model, train_data, validation_data, test_data = train_pipeline_common(
        pj, input_data, training_horizons, test_fraction=0.15, backtest=True
    )

    # Predict
    model_forecast = model.predict(test_data.iloc[:, 1:-1])
    forecast = pd.DataFrame(index=test_data.index, data={"forecast": model_forecast})

    # Define tAhead to something meaningfull in the context of a backtest
    forecast["tAhead"] = test_data.iloc[:, -1]

    # Add confidence
    forecast = ConfidenceIntervalApplicator(
        model, test_data.iloc[:, 1:-1]
    ).add_confidence_interval(forecast, pj)

    # Prepare for output
    forecast = add_prediction_job_properties_to_forecast(
        pj, forecast, algorithm_type="backtest"
    )

    # Add column with realised load and horizon information
    forecast["realised"] = test_data.iloc[:, 0]
    forecast["horizon"] = test_data.iloc[:, -1]

    return forecast, model, train_data, validation_data, test_data

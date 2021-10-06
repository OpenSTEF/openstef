# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
from datetime import datetime, timedelta
from typing import Tuple

import numpy as np
import sklearn.metrics as metrics


def generate_forecast_datetime_range(
        resolution_minutes: int, horizon_minutes: int
) -> Tuple[datetime, datetime]:
    # get current date and time UTC
    datetime_utc = datetime.utcnow()
    # Datetime range for time interval to be predicted
    forecast_start = datetime_utc - timedelta(minutes=resolution_minutes)
    forecast_end = datetime_utc + timedelta(minutes=horizon_minutes)

    return forecast_start, forecast_end


def get_metrics(y_pred: np.array, y_true: np.array) -> dict:
    """ Calculate the metrics for a prediction

    Args:
        y_pred: np.array
        y_true: np.array

    Returns:
        dictionary: metrics for the prediction
    """
    explained_variance = metrics.explained_variance_score(y_true, y_pred)
    mean_absolute_error = metrics.mean_absolute_error(y_true, y_pred)
    mse = metrics.mean_squared_error(y_true, y_pred)
    r2 = metrics.r2_score(y_true, y_pred)
    return {
        "explained_variance": explained_variance,
        "r2": r2,
        "MAE": mean_absolute_error,
        "MSE": mse,
        "RMSE": np.sqrt(mse)}

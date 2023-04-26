# SPDX-FileCopyrightText: 2017-2022 Contributors to the OpenSTEF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

# This file contains the Objective loss functions for quantile regression from:
# https://gist.github.com/Nikolay-Lysenko/06769d701c1d9c9acb9a66f2f9d7a6c7
#
# SPDX-FileCopyrightText: 2017 Nikolay Lysenko
#
# SPDX-License-Identifier: MIT
"""This module contains all metrics to assess forecast quality."""
from typing import Callable

import numpy as np
import pandas as pd
import xgboost


def get_eval_metric_function(metric_name: str) -> Callable:
    """Gets a metric if it is available.

    Args:
        metric_name: Name of the metric.

    Returns:
        Function to calculate the metric.

    """
    evaluation_function = {
        "rmse": rmse,
        "bias": bias,
        "nsme": nsme,
        "mae": mae,
        "r_mae": r_mae,
        "frac_in_stdev": frac_in_stdev,
        "r_mae_highest": r_mae_highest,
        "r_mne_highest": r_mne_highest,
        "r_mpe_highest": r_mpe_highest,
        "r_mae_lowest": r_mae_lowest,
        "skill_score": skill_score,
        "skill_score_positive_peaks": skill_score_positive_peaks,
        "franks_skill_score": franks_skill_score,
        "franks_skill_score_peaks": franks_skill_score_peaks,
    }.get(metric_name, None)

    if evaluation_function is None:
        raise KeyError(f"Unknown evaluation metric function {metric_name}")

    return evaluation_function


def rmse(realised: pd.Series, forecast: pd.Series) -> float:
    """Function that calculates the Root Mean Square Error based on the true and prediciton.

    Args:
        realised: Realised load.
        forecast: Forecasted load.

    Returns:
        Root Mean Square Error

    """
    return np.sqrt(((realised - forecast) ** 2).mean())


def bias(realised: pd.Series, forecast: pd.Series) -> float:
    """Function that calculates the absolute bias in % based on the true and prediciton.

    Args:
        realised: Realised load.
        forecast: Forecasted load.

    Returns:
        Bias

    """
    return np.mean(forecast - realised)


def nsme(realised: pd.Series, forecast: pd.Series) -> float:
    """Function that calculates the Nash-sutcliffe model efficiency based on the true and prediciton.

    Args:
        realised: Realised load.
        forecast: Forecasted load.

    Returns:
        Nash-sutcliffe model efficiency

    """
    try:
        return 1 - sum((forecast - realised) ** 2) / sum(
            (realised - np.mean(realised)) ** 2
        )
    except ZeroDivisionError:  # means the error is 0
        return 1


def mae(realised: pd.Series, forecast: pd.Series) -> float:
    """Function that calculates the mean absolute error based on the true and prediction."""
    return np.mean(np.abs(forecast - realised))


def r_mae(realised: pd.Series, forecast: pd.Series) -> float:
    """Function that calculates the relative mean absolute error based on the true and prediction.

    The range is based on the load range of the previous two weeks

    """
    # Determine load range on entire dataset
    range_ = (
        realised.max() - realised.min()
        if (realised.max() - realised.min()) != 0
        else np.nan
    )

    return mae(realised, forecast) / range_


def frac_in_stdev(realised: pd.Series, forecast: pd.Series, stdev: pd.Series) -> float:
    """Function that calculates the amount of measurements that are within one stdev of our predictions."""
    outside_stdev = forecast[(forecast - realised).abs() > stdev]
    return round((1 - (len(outside_stdev) / len(forecast))), 2)


def r_mae_highest(
    realised: pd.Series, forecast: pd.Series, percentile: float = 0.95
) -> float:
    """Function that calculates the relative mean absolute error for the 5 percent highest realised values.

    The range is based on the load range of the previous two weeks.

    """
    # Check if length of both arrays is equal
    if len(np.array(realised)) != len(np.array(forecast)):
        raise ValueError(
            "Error metric can only be calculated for arrays of equal length!"
        )

    # Determine load range on entire dataset
    range_ = (
        realised.max() - realised.min()
        if (realised.max() - realised.min()) != 0
        else np.nan
    )

    # Get highest percentile of values
    highest_values = realised > np.percentile(realised, percentile)

    # Calculate mae
    r_mae_highest = mae(realised[highest_values], forecast[highest_values]) / range_

    return r_mae_highest


def r_mne_highest(realised: pd.Series, forecast: pd.Series) -> float:
    """Function that calculates the relative mean negative error for the 5 percent highest realised values.

    The range is based on the load range of the previous two weeks, this measure quantifies how much we underestimate
    peaks.

    """
    # Combine series in one DataFrame
    combined = pd.concat([realised, forecast], axis=1)

    # Determine load range on entire dataset
    range_ = (
        combined["load"].max() - combined["load"].min()
        if (combined["load"].max() - combined["load"].min()) != 0
        else np.nan
    )

    # Select 5 percent highest realised load values
    combined["highest"] = combined["load"][
        combined["load"] > combined["load"].quantile(0.95)
    ]
    combined = combined[np.invert(np.isnan(combined["highest"]))]

    # Calculate rMNE for the selected points
    diff = combined[forecast.name] - combined["load"]

    if len(diff[diff < 0]) < 2:
        return 0.0

    r_mne_highest = np.mean(diff[diff < 0]) / range_

    if np.isnan(r_mne_highest):
        return 99999.0

    return r_mne_highest


def r_mpe_highest(realised: pd.Series, forecast: pd.Series) -> float:
    """Function that calculates the relative mean positive error for the 5 percent highest realised values.

    The range is based on the load range of the previous two weeks, this measure quantifies how much we overestimate
    peaks.

    """
    # Combine series in one DataFrame
    combined = pd.concat([realised, forecast], axis=1)

    # Determine load range on entire dataset
    range_ = (
        combined["load"].max() - combined["load"].min()
        if (combined["load"].max() - combined["load"].min()) != 0
        else np.nan
    )

    # Select 5 percent highest realised load values
    combined["highest"] = combined["load"][
        combined["load"] > combined["load"].quantile(0.95)
    ]
    combined = combined[np.invert(np.isnan(combined["highest"]))]

    # Calculate rMPE for the selected points

    diff = combined[forecast.name] - combined["load"]

    if len(diff[diff > 0]) < 2:
        return 0.0

    r_mpe_highest = np.mean(diff[diff > 0]) / range_

    if np.isnan(r_mpe_highest):
        return 99999.0
    return r_mpe_highest


def r_mae_lowest(
    realised: pd.Series, forecast: pd.Series, quantile: float = 0.05
) -> float:
    """Function that calculates the relative mean absolute error for the 5 percent lowest realised values.

    The range is based on the load range of the previous two weeks.

    """
    # Determine load range on entire dataset
    range_ = (
        realised.max() - realised.min()
        if (realised.max() - realised.min()) != 0
        else np.nan
    )

    # Get lowest percentile of values
    lowest_values = realised < np.quantile(realised, quantile)
    # Calculate mae
    r_mae_lowest = mae(realised[lowest_values], forecast[lowest_values]) / range_

    return r_mae_lowest


def skill_score(realised: pd.Series, forecast: pd.Series, mean: pd.Series) -> float:
    """Function that calculates the skill score.

    Thise indicates model performance relative to a reference, in this case the mean of the realised values. The range
    is based on the load range of the previous two weeks.

    """
    combined = pd.concat([realised, forecast], axis=1)
    combined["mean"] = mean

    skill_score = 1 - (mae(realised, forecast) / mae(realised, combined["mean"]))

    if np.isnan(skill_score):
        return 0

    return skill_score


def skill_score_positive_peaks(
    realised: pd.Series, forecast: pd.Series, mean: pd.Series
) -> float:
    """Calculates skill score on positive peaks."""
    # Combine series in one DataFrame
    combined = pd.concat([realised, forecast], axis=1)

    # Select 5 percent highest realised load values
    combined["highest"] = combined["load"][
        combined["load"] > combined["load"].quantile(0.95)
    ]
    combined = combined[np.invert(np.isnan(combined["highest"]))]

    # Calculate rMAE for the selected points
    skill_score_highest = skill_score(combined["load"], combined[forecast.name], mean)

    if np.isnan(skill_score_highest):
        return 0

    return skill_score_highest


def franks_skill_score(
    realised: pd.Series, forecast: pd.Series, basecase: pd.Series, range_: float = 1.0
) -> float:
    """Calculate Franks skill score."""
    # Combine series in one DataFrame
    combined = pd.concat([realised, forecast], axis=1)
    if range_ == 1.0:
        range_ = (
            combined["load"].max() - combined["load"].min()
            if (combined["load"].max() - combined["load"].min()) != 0
            else np.nan
        )

    franks_skill_score = (mae(realised, basecase) - mae(realised, forecast)) / range_

    if np.isnan(franks_skill_score):
        return 0

    return franks_skill_score


def franks_skill_score_peaks(
    realised: pd.Series, forecast: pd.Series, basecase: pd.Series
) -> float:
    """Calculate Franks skill score on positive peaks."""
    # Combine series in one DataFrame
    combined = pd.concat([realised, forecast, basecase], axis=1)

    range_ = (
        combined["load"].max() - combined["load"].min()
        if (combined["load"].max() - combined["load"].min()) != 0
        else np.nan
    )
    # Select 5 percent highest realised load values
    combined["highest"] = combined["load"][
        combined["load"] > combined["load"].quantile(0.95)
    ]
    combined = combined[np.invert(np.isnan(combined["highest"]))]

    # Calculate rMAE for the selected points
    franks_skill_score_highest = franks_skill_score(
        combined["load"],
        combined[forecast.name],
        combined[basecase.name],
        range_=range_,
    )

    if np.isnan(franks_skill_score_highest):
        return 0

    return franks_skill_score_highest


# Objective loss functions for quantile regression, from: https://gist.github.com/Nikolay-Lysenko/06769d701c1d9c9acb9a66f2f9d7a6c7

# SPDX-FileCopyrightText: 2017 Nikolay Lysenko
#
# SPDX-License-Identifier: MIT


def xgb_quantile_eval(
    preds: np.ndarray, dmatrix: xgboost.DMatrix, quantile: float = 0.2
) -> str:
    """Customized evaluational metric that equals to quantile regression loss (also known as pinball loss).

    Quantile regression is regression that estimates a specified quantile of target's distribution conditional on given features.

    Args:
        preds: Predicted values
        dmatrix: xgboost.DMatrix of the input data.
        quantile: Target quantile.

    Returns:
        Loss information


    # See also:
    https://gist.github.com/Nikolay-Lysenko/06769d701c1d9c9acb9a66f2f9d7a6c7

    """
    labels = dmatrix.get_label()
    return (
        "q{}_loss".format(quantile),
        np.nanmean(
            (preds >= labels) * (1 - quantile) * (preds - labels)
            + (preds < labels) * quantile * (labels - preds)
        ),
    )


def xgb_quantile_obj(
    preds: np.ndarray, dmatrix: xgboost.DMatrix, quantile: float = 0.2
) -> tuple[np.ndarray, np.ndarray]:
    """Quantile regression objective fucntion.

    Computes first-order derivative of quantile regression loss and a non-degenerate substitute for second-order
    derivative.

    Substitute is returned instead of zeros, because XGBoost requires non-zero second-order derivatives. See
    this page: https://github.com/dmlc/xgboost/issues/1825 to see why it is possible to use this trick. However, be sure
    that hyperparameter named `max_delta_step` is small enough to satisfy:``0.5 * max_delta_step <= min(quantile, 1 - quantile)``.

    Args:
        preds: numpy.ndarray
        dmatrix: xgboost.DMatrix
        quantile: float

    Returns:
        Gradient and Hessian

    # See also:
    https://gist.github.com/Nikolay-Lysenko/06769d701c1d9c9acb9a66f2f9d7a6c7

    Reasoning for the hessian:
    https://gist.github.com/Nikolay-Lysenko/06769d701c1d9c9acb9a66f2f9d7a6c7#gistcomment-2322558

    """
    try:
        assert 0 <= quantile <= 1
    except AssertionError:
        raise ValueError("Quantile value must be float between 0 and 1.")

    labels = dmatrix.get_label()
    errors = preds - labels

    left_mask = errors < 0
    right_mask = errors > 0

    # The factor `* errors` is different from the original implementation, however
    # this addition makes the objective function scalable with the size of the error.
    # This solves issues with regression on large (>100) input data.
    grad = (quantile * left_mask + (1 - quantile) * right_mask) * errors
    hess = np.ones_like(preds)

    return grad, hess

# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Utility functions for calculating pinball loss errors.

This module provides a function to compute the pinball loss for quantile regression.
"""

import pandas as pd


def calculate_pinball_errors(y_true: pd.Series, y_pred: pd.Series, alpha: float) -> pd.Series:
    """Calculate pinball loss for given true and predicted values.

    Args:
        y_true: True values as a pandas Series.
        y_pred: Predicted values as a pandas Series.
        alpha: Quantile value.

    Returns:
        A pandas Series containing the pinball loss for each sample.
    """
    diff = y_true - y_pred
    sign = (diff >= 0).astype(float)
    return alpha * sign * diff - (1 - alpha) * (1 - sign) * diff

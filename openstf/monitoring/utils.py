# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

import datetime

import pandas as pd


def check_max_ahead(pred, max_ahead):
    """Check if the values of pred extend max_ahead hours into the future
    Input:
        - pred: pd.DataFrame(columns=['datetimeFC', *])
        - max_ahead: float, time ahead in hours

    Return:
        - True / False
    """

    # get prediction max ahead
    pred_max_ahead = (
        pd.to_datetime(pred.datetimeFC).max().tz_convert(None)
        - datetime.datetime.utcnow()
    )

    # check max ahead
    if pred_max_ahead.total_seconds() / 60.0 / 60.0 < max_ahead:
        return True
    return False


def check_datetimes_unique(pred):
    """Check if the values of pred only have unique datetimes
    Input:
        - pred: pd.DataFrame(columns=['datetimeFC', *])

    Return:
        - True / False
    """
    # get datetimes
    datetimes = pred.datetimeFC

    # check uniqueness of datetimes
    if datetimes.duplicated().any():
        return True
    return False

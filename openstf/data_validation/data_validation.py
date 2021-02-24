# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

from datetime import timedelta

import numpy as np
from ktpbase.log import logging

from openstf.feature_engineering import general


def validate(data):
    logger = logging.get_logger(__name__)
    # Drop 'false' measurements. e.g. where load appears to be constant.
    threshold = 6 * 4  # number of repeated values
    data = general.nan_repeated(data, max_length=threshold, column_name=data.columns[0])
    num_const_load_values = len(data) - len(data.iloc[:, 0].dropna())
    logger.debug(
        f"Changed {num_const_load_values} values of constant load to NA.",
        num_const_load_values=num_const_load_values,
    )
    return data


def clean(data):
    logger = logging.get_logger(__name__)
    data = data[data.index.min() + timedelta(weeks=2) :]
    len_original = len(data)
    # TODO Look into this
    # Remove where load is NA # # df.dropna?
    data = data.loc[np.isnan(data.iloc[:, 0]) != True, :]  # noqa E712
    num_removed_values = len_original - len(data)
    logger.debug(
        f"Removed {num_removed_values} NA values", num_removed_values=num_removed_values
    )
    return data

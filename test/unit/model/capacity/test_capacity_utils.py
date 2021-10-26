# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

import unittest
from datetime import datetime, timedelta

import numpy as np
import pytz

from openstf.feature_engineering.capacity_prognoses_features import (
    apply_capacity_features,
)
from openstf.model.capacity import utils
from test.utils import BaseTestCase
from test.utils.data import TestData

data = TestData.load("input_data_train.pickle")
load_profile_names = TestData.load("input_data_train_load_profile_names.json")


class TestCapacityPrognosisUtils(BaseTestCase):
    def test_happy_prepare_training_data(self):
        feature_data, _ = apply_capacity_features(
            data, y_col="load_max", y_hor=[1, 2], load_profile_names=load_profile_names
        )
        train_x, train_y, train_h, val_x, val_y, val_h = utils.prepare_training_data(
            df=feature_data, y_col="load_max", val_n=4, val_width="14D"
        )

    def test_happy_prepare_prediction_data(self):
        # move data to present and set load for last 14 days to zero
        pc_input = data.copy(deep=True)
        pc_input[pc_input.index.max() - timedelta(days=14) :]["load"] = np.nan
        offset = datetime.utcnow().replace(tzinfo=pytz.utc) - (
            data.index.max() - timedelta(days=14)
        )
        pc_input.index = pc_input.index + offset

        feature_data, _ = apply_capacity_features(
            pc_input,
            y_col="load_max",
            y_hor=[1, 2],
            load_profile_names=load_profile_names,
        )
        x = utils.prepare_prediction_data(
            df=feature_data, y_col="load_max", y_hor=[1, 2]
        )
        return x


if __name__ == "__main__":
    unittest.main()

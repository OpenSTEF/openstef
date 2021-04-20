# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
import unittest

from test.utils.data import TestData

from openstf.feature_engineering.lag_features import extract_lag_features
from openstf.enums import MLModelType
from openstf.model.serializer.creator import ModelSerializerCreator

from test.utils import BaseTestCase


class TestGeneralExtractMinuteFeatures(BaseTestCase):
    def setUp(self):
        super().setUp()
        serializer_creator = ModelSerializerCreator()
        serializer = serializer_creator.create_model_serializer(MLModelType("xgb"))
        model_folder = TestData.TRAINED_MODELS_FOLDER / "307"
        self.model, model_file = serializer.load(307, model_folder)

    def test_extract_minute_features(self):
        testlist_minutes, testlist_days = extract_lag_features(self.model.feature_names)
        self.assertEqual(
            testlist_minutes,
            [
                900,
                780,
                15,
                1425,
                660,
                540,
                30,
                420,
                1320,
                300,
                45,
                1200,
                2865,
                180,
                1080,
                60,
                960,
                840,
                720,
                600,
                480,
                1380,
                360,
                1260,
                240,
                1140,
                120,
                1020,
            ],
        )
        self.assertEqual(testlist_days, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])


if __name__ == "__main__":
    unittest.main()

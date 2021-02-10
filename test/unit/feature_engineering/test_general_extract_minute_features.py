# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0
import unittest

from test.utils.data import TestData

from stf.feature_engineering.general import extract_minute_features
from stf.model.general import MLModelType
from stf.model.serializer.creator import ModelSerializerCreator

from test.utils import BaseTestCase


class TestGeneralExtractMinuteFeatures(BaseTestCase):

    def setUp(self):
        serializer_creator = ModelSerializerCreator()
        serializer = serializer_creator.create_model_serializer(MLModelType('xgb'))
        model_folder = TestData.TRAINED_MODELS_FOLDER
        self.model, model_file = serializer.load(123, model_folder)

    def test_extract_minute_features(self):
        testlist = extract_minute_features(self.model.feature_names)
        self.assertEqual(testlist, [
            900, 780, 15, 1425, 660, 540, 30, 420, 1320, 300, 45, 1200, 2865, 180, 1080,
            60, 960, 840, 720, 600, 480, 1380, 360, 1260, 240, 1140, 120, 1020
        ])


if __name__ == "__main__":
    unittest.main()

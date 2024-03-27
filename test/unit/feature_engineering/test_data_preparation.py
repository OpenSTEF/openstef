# SPDX-FileCopyrightText: 2017-2023 Contributors to the OpenSTEF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
from pathlib import Path
from test.unit.utils.data import TestData
from unittest import TestCase
from unittest.mock import MagicMock, patch

import numpy as np

from openstef.feature_engineering.data_preparation import LegacyDataPreparation
from openstef.model.serializer import MLflowSerializer


class TestDataPreparation(TestCase):
    @patch("openstef.model.serializer.MLflowSerializer._get_model_uri")
    def setUp(self, _get_model_uri_mock) -> None:
        self.pj, self.model_specs = TestData.get_prediction_job_and_modelspecs(pid=307)
        self.input_data = TestData.load("input_data.csv")
        self.input_data.iloc[-5:, 0] = np.nan

        self.model = MagicMock()
        self.model.feature_names = ["apx"]

    def test_legacy_prepare_forecast_no_model(self):
        # Test the error if no model is provided for forecast
        data_prep = LegacyDataPreparation(self.pj, self.model_specs)

        with self.assertRaises(ValueError):
            data_prep.prepare_forecast_data(self.input_data[["load"]])

    def test_legacy_prepare_forecast_happy_flow(self):
        # Test for expected column order of the output
        # Also check "horizons" is not in the output
        features = self.model.feature_names

        data_with_features, _ = LegacyDataPreparation(
            self.pj, self.model_specs, self.model, horizons=[0.25]
        ).prepare_forecast_data(self.input_data[["load"]])

        self.assertListEqual(
            list(np.sort(features)), list(np.sort(data_with_features.columns.to_list()))
        )

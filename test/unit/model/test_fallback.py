# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

import unittest
from datetime import datetime, timedelta, timezone
import numpy as np
from test.utils import BaseTestCase, TestData
from unittest import mock
from unittest.mock import MagicMock, patch

from openstf.model.fallback import generate_fallback


class TestFallback(BaseTestCase):
    def test_fallback_happy(self):
        """Test happy flow of fallback forecast"""
        expected_forecast = TestData.load("fallback_forecast.csv")
        load = TestData.load("fallback_load.csv")
        forc_section = TestData.load("fallback_index.csv")

        fallback_forecast = generate_fallback(
            forecast_input=forc_section.index,
            load=load,
            fallback_strategy='extreme_day'
        )

        self.assertDataframeEqual(fallback_forecast, expected_forecast)

    def test_empty_load(self):
        """Test if exception is raised if load is empty"""
        load = TestData.load("fallback_load.csv")
        load *= np.nan
        forc_section = TestData.load("fallback_index.csv")
        self.assertRaises(ValueError,
                          generate_fallback,
                            forecast_input=forc_section.index,
                            load=load,
                          )

    def test_unimplemented_input(self):
        load = TestData.load("fallback_load.csv")
        forc_section = TestData.load("fallback_index.csv")
        self.assertRaises(NotImplementedError,
                          generate_fallback,
                            forecast_input=forc_section.index,
                            load=load,
                              fallback_strategy='SomeWeirdNotImplementedStrategy'

                          )



if __name__ == "__main__":
    unittest.main()
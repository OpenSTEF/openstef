# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

import unittest
from datetime import datetime, timedelta, timezone
from test.utils import BaseTestCase, TestData
from unittest import mock
from unittest.mock import MagicMock, patch

from openstf.model.fallback import generate_fallback

class TestFallback(BaseTestCase):
    def test_fallback_happy(self):
        """Test happy flow of fallback forecast"""
        # Load input data
        forecast = TestData.load("fallback_forecast.csv")
        load = TestData.load("fallback_load.csv")
        forc_section = TestData.load("fallback_index.csv")

        fallback_forecast = generate_fallback(
            forecast_input=forc_section.index,
            load=load,
            kind='extreme_day'
        )

        self.assertDataframeEqual(fallback_forecast, forecast)

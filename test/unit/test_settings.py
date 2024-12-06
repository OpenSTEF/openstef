# SPDX-FileCopyrightText: 2017-2023 Contributors to the OpenSTEF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
import os
from unittest import TestCase

from openstef.app_settings import AppSettings


class TestAppSettings(TestCase):
    def test_create_forecast_task_parsing_weather_sources_env(self):
        """Test happy flow of create forecast task."""
        # Arrange
        os.environ[
            "OPENSTEF_WEATHER_SOURCES"
        ] = '["weather_source_1","weather_source_2"]'

        # Act
        settings = AppSettings()

        # Assert
        self.assertEqual(
            settings.weather_sources, ["weather_source_1", "weather_source_2"]
        )

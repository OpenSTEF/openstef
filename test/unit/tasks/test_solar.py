# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

import unittest
from unittest.mock import MagicMock

from openstf.tasks.create_solar_forecast import make_solar_predicion_pj
from openstf_dbc.services.prediction_job import PredictionJobDataClass

from test.utils import BaseTestCase, TestData


class TestSolar(BaseTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.test_solar_input = TestData.load("solar_input.csv")
        self.solar_ref = TestData.load("solar_ref.csv")
        self.pj = {
            "id": 71,
            "typ": "solar",
            "model": "latest",
            "horizon_minutes": 2880,
            "resolution_minutes": 15,
            "name": "Provincies",
            "lat": 52.5,
            "lon": 4.9,
            "sid": None,
            "radius": 30,
            "peak_power": 180961000.0,
            "description": "",
        }

    def test_make_solar_predicion_pj(self):

        context = MagicMock()
        context.database.get_solar_input = MagicMock(return_value=self.test_solar_input)

        make_solar_predicion_pj(self.pj, context)

        self.assertTrue(context.logger.info.called)
        self.assertTrue(context.database.write_forecast.called)
        refference_result = context.database.write_forecast.call_args
        self.assertEqual(
            refference_result[0][0].columns.all(), self.solar_ref.columns.all()
        )
        self.assertEqual(len(refference_result[0][0]), len(self.solar_ref))


if __name__ == "__main__":
    unittest.main()

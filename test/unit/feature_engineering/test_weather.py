# SPDX-FileCopyrightText: 2017-2023 Contributors to the OpenSTEF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

import unittest
from test.unit.utils.base import BaseTestCase

import numpy as np
import pandas as pd

from openstef.feature_engineering import weather_features
from openstef.enums import WeatherConstants


class HumidityCalculationsTest(BaseTestCase):
    def test_good_input(self):
        temp = 40
        rh = 0.5
        pressure = 101300
        expected = {
            WeatherConstants.SATURATION_PRESSURE: 74.0413358265025,
            WeatherConstants.VAPOUR_PRESSURE: 37.02066791325125,
            "dewpoint": 27.606507491812938,
            "air_density": 1.0184141379792693,
        }
        result = weather_features.humidity_calculations(temp, rh, pressure)

        self.assertEqual(result.keys(), expected.keys())
        np.testing.assert_allclose(list(result.values()), list(expected.values()))

    def test_good_input_high_RH(self):
        temp = 40
        rh = 50
        pressure = 101300
        expected = {
            WeatherConstants.SATURATION_PRESSURE: 74.0413358265025,
            WeatherConstants.VAPOUR_PRESSURE: 37.02066791325125,
            "dewpoint": 27.606507491812938,
            "air_density": 1.0184141379792693,
        }
        result = weather_features.humidity_calculations(temp, rh, pressure)

        self.assertEqual(result.keys(), expected.keys())
        np.testing.assert_allclose(list(result.values()), list(expected.values()))

    def test_str_input(self):
        temp = "40"
        rh = 50
        pressure = 101300
        with self.assertRaises(TypeError):
            weather_features.humidity_calculations(temp, rh, pressure)

    def test_pd_input(self):
        dict_for_df = {
            WeatherConstants.TEMPERATURE: {
                0: 17.527984619140625,
                1: 18.406951904296875,
                2: 19.725128173828125,
                3: 21.138153076171875,
                4: 22.027984619140625,
                5: 22.796356201171875,
                6: 21.144195556640625,
                7: 20.615020751953125,
                8: 21.490631103515625,
                9: 21.419891357421875,
                10: 20.398956298828125,
            },
            "humidity": {
                0: 0.7382869124412537,
                1: 0.7526984214782715,
                2: 0.7334562540054321,
                3: 0.6899152845144272,
                4: 0.6502455472946167,
                5: 0.6159644275903702,
                6: 0.7110961973667145,
                7: 0.7494502067565918,
                8: 0.7257296442985535,
                9: 0.6748714596033096,
                10: 0.6509994566440582,
            },
            "pressure": {
                0: 101683.453125,
                1: 101669.109375,
                2: 101660.3203125,
                3: 101641.5234375,
                4: 101621.359375,
                5: 101579.0859375,
                6: 101606.0546875,
                7: 101571.265625,
                8: 101525.65625,
                9: 101544.28125,
                10: 101571.5859375,
            },
            WeatherConstants.SATURATION_PRESSURE: {
                0: 20.060412049531564,
                1: 21.201730117304287,
                2: 23.020084506148535,
                3: 25.119690610620534,
                4: 26.526298638749843,
                5: 27.79588978764522,
                6: 25.129017560585353,
                7: 24.323605484298096,
                8: 25.668859869681906,
                9: 25.557810225856993,
                10: 24.00131027167236,
            },
            WeatherConstants.VAPOUR_PRESSURE: {
                0: 14.81033967434798,
                1: 15.958508791903265,
                2: 16.884224948768193,
                3: 17.33045849454065,
                4: 17.248607576054336,
                5: 17.121279342411903,
                6: 17.869148830893636,
                7: 18.229331159272977,
                8: 18.628652542773665,
                9: 17.248236691388502,
                10: 15.62483994560416,
            },
            "dewpoint": {
                0: 12.8122960368354,
                1: 13.956688695601496,
                2: 14.827795599027679,
                3: 15.232803076321495,
                4: 15.159205631990902,
                5: 15.044104419011173,
                6: 15.709695359586199,
                7: 16.021562280746632,
                8: 16.361055609346657,
                9: 15.15887144798449,
                10: 13.631785491225672,
            },
            "air_density": {
                0: 1.1013913267467517,
                1: 1.0979113262307172,
                2: 1.0928715770331208,
                3: 1.087421233686219,
                4: 1.0839283775352135,
                5: 1.080664903363787,
                6: 1.0870172439232104,
                7: 1.0886010055207498,
                8: 1.0848768927921209,
                9: 1.085342086995991,
                10: 1.0894162659923423,
            },
        }
        df = pd.DataFrame.from_dict(dict_for_df)
        humidity_df = weather_features.humidity_calculations(
            df.temp, df.humidity, df.pressure
        )
        result_df = df[
            [WeatherConstants.SATURATION_PRESSURE, WeatherConstants.VAPOUR_PRESSURE, "dewpoint", "air_density"]
        ]
        self.assertDataframeEqual(humidity_df, result_df)


class RadiationCalculationsTest(BaseTestCase):
    def setUp(self):
        # Arrange some mock radiation data. 60 datapoints with high and low radiation.
        times = pd.DatetimeIndex(
            [
                "2020-08-20 10:00:00+00:00",
                "2020-08-20 10:15:00+00:00",
                "2020-08-20 10:30:00+00:00",
                "2020-08-20 19:00:00+00:00",
                "2020-08-20 19:15:00+00:00",
                "2020-08-20 19:30:00+00:00",
                "2020-08-20 19:45:00+00:00",
                "2020-08-20 20:00:00+00:00",
                "2020-08-20 20:15:00+00:00",
                "2020-08-20 20:30:00+00:00",
                "2020-08-20 20:45:00+00:00",
                "2020-08-20 21:00:00+00:00",
                "2020-08-20 21:15:00+00:00",
                "2020-08-20 21:30:00+00:00",
                "2020-08-20 21:45:00+00:00",
                "2020-08-20 22:00:00+00:00",
                "2020-08-20 22:15:00+00:00",
                "2020-08-20 22:30:00+00:00",
                "2020-08-20 22:45:00+00:00",
                "2020-08-20 23:00:00+00:00",
                "2020-08-20 23:15:00+00:00",
                "2020-08-20 23:30:00+00:00",
                "2020-08-20 23:45:00+00:00",
                "2020-08-21 00:00:00+00:00",
                "2020-08-21 00:15:00+00:00",
                "2020-08-21 00:30:00+00:00",
                "2020-08-21 00:45:00+00:00",
                "2020-08-21 01:00:00+00:00",
                "2020-08-21 01:15:00+00:00",
                "2020-08-21 04:15:00+00:00",
                "2020-08-21 04:30:00+00:00",
                "2020-08-21 04:45:00+00:00",
                "2020-08-21 05:00:00+00:00",
                "2020-08-21 05:15:00+00:00",
                "2020-08-21 05:30:00+00:00",
                "2020-08-21 05:45:00+00:00",
                "2020-08-21 06:00:00+00:00",
                "2020-08-21 06:15:00+00:00",
                "2020-08-21 06:30:00+00:00",
                "2020-08-21 06:45:00+00:00",
                "2020-08-21 07:00:00+00:00",
                "2020-08-21 07:15:00+00:00",
                "2020-08-21 07:30:00+00:00",
                "2020-08-21 07:45:00+00:00",
                "2020-08-21 08:00:00+00:00",
                "2020-08-21 08:15:00+00:00",
                "2020-08-21 08:30:00+00:00",
                "2020-08-21 08:45:00+00:00",
                "2020-08-21 09:00:00+00:00",
                "2020-08-21 09:15:00+00:00",
                "2020-08-21 09:30:00+00:00",
                "2020-08-21 09:45:00+00:00",
                "2020-08-21 10:00:00+00:00",
                "2020-08-21 10:15:00+00:00",
                "2020-08-21 10:30:00+00:00",
                "2020-08-21 19:00:00+00:00",
                "2020-08-21 19:15:00+00:00",
                "2020-08-21 19:30:00+00:00",
                "2020-08-21 19:45:00+00:00",
                "2020-08-21 20:00:00+00:00",
            ]
        )
        self.radiation = pd.DataFrame(
            [
                1085511.0,
                986250.0,
                886989.0,
                -0.2421875,
                4.10546875,
                8.453125,
                6.33984375,
                4.2265625,
                2.11328125,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.24e-11,
                2.47e-11,
                3.71e-11,
                4.95e-11,
                4.95e-11,
                4.95e-11,
                4.95e-11,
                4.95e-11,
                4.95e-11,
                4.95e-11,
                4.95e-11,
                146385.4375,
                194289.25,
                268365.6563,
                342442.0625,
                416518.4688,
                490594.875,
                522269.2188,
                553943.5625,
                585617.9063,
                617292.25,
                652557.5,
                687822.75,
                723088.0,
                758353.25,
                1116257.688,
                1474162.125,
                1832066.563,
                2189971.0,
                2273495.375,
                2357019.75,
                2440544.125,
                2524068.5,
                2559136.375,
                2594204.25,
                2629272.125,
                2664340.0,
                3.2421875,
                0.31640625,
                -2.609375,
                -1.95703125,
                -1.3046875,
            ],
            index=times,
            columns=[WeatherConstants.RADIATION],
        )  # .sort_index()
        self.pj = {"lon": 52.98749605, "lat": 6.288165514}

    def test_dni(self):
        # Act
        data = weather_features.add_additional_solar_features(self.radiation, self.pj)

        # Assert
        self.assertIn("dni", data.columns, msg="dni was not added as a feature.")
        self.assertEqual(data["dni"].isnull().sum(), 0)

    def test_global_tilted_irradiance(self):
        # Act
        data = weather_features.add_additional_solar_features(self.radiation, self.pj)

        # Assert
        self.assertIn("gti", data.columns, msg="gti was not added as a feature.")
        self.assertEqual(data["gti"].isnull().sum(), 0)


if __name__ == "__main__":
    unittest.main()

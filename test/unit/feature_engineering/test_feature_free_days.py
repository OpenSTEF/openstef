# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
import unittest

from stf.feature_engineering.feature_free_days import create_holiday_functions

from test.utils import BaseTestCase

expected_keys = ['is_national_holiday', 'is_nieuwjaarsdag', 'is_goede_vrijdag',
                 'is_eerste_paasdag', 'is_tweede_paasdag', 'is_koningsdag',
                 'is_bevrijdingsdag', 'is_bridgedaybevrijdingsdag', 'is_hemelvaart',
                 'is_bridgedayhemelvaart', 'is_eerste_pinksterdag',
                 'is_tweede_pinksterdag', 'is_eerste_kerstdag', 'is_tweede_kerstdag',
                 'is_bridgedaykoningsdag', 'is_bridgeday', 'is_schoolholiday',
                 'is_bouwvakmidden', 'is_kerstvakantie', 'is_voorjaarsvakantiemidden',
                 'is_voorjaarsvakantiezuid', 'is_bouwvakzuid', 'is_zomervakantienoord',
                 'is_zomervakantiemidden', 'is_voorjaarsvakantienoord',
                 'is_zomervakantiezuid', 'is_herfstvakantiemidden', 'is_bouwvaknoord',
                 'is_herfstvakantienoord', 'is_meivakantie', 'is_herfstvakantiezuid']


class GeneralTest(BaseTestCase):

    def test_create_holiday_functions(self):
        holiday_functions = create_holiday_functions(country="NL")

        print(holiday_functions.keys())

        # Assert for every holiday a function is available and no extra functions are generated # noqa E501>
        self.assertEqual(all([key in holiday_functions.keys()
                              for key in expected_keys]), True)
        self.assertEqual(
            all([key in expected_keys for key in holiday_functions.keys()]), True)


if __name__ == "__main__":
    unittest.main()

# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

from stf.feature_engineering.feature_free_days import create_holiday_functions

from test.utils import BaseTestCase

expected_keys = ['IsNationalHoliday', 'IsNieuwjaarsdag', 'IsGoede Vrijdag', 'IsEerste paasdag', 'IsTweede paasdag', 'IsKoningsdag', 'IsBevrijdingsdag', 'IsBridgedayBevrijdingsdag', 'IsHemelvaart', 'IsBridgedayHemelvaart', 'IsEerste Pinksterdag', 'IsTweede Pinksterdag', 'IsEerste Kerstdag', 'IsTweede Kerstdag', 'IsBridgedayKoningsdag', 'IsSchoolholiday', 'IsVoorjaarsvakantieNoord', 'IsHerfstvakantieZuid', 'IsZomervakantieZuid', 'IsHerfstvakantieMidden', 'IsVoorjaarsvakantieMidden', 'IsBouwvakNoord', 'IsMeivakantie', 'IsHerfstvakantieNoord', 'IsBouwvakMidden', 'IsVoorjaarsvakantieZuid', 'IsZomervakantieMidden', 'IsZomervakantieNoord', 'IsKerstvakantie', 'IsBouwvakZuid']

class GeneralTest(BaseTestCase):

    def test_create_holiday_functions(self):

        holiday_functions=create_holiday_functions(country = "NL")

        print(holiday_functions.keys())

        # Assert for every holiday a function is available and no extra functions are generated
        self.assertEqual(all([key in holiday_functions.keys()
                              for key in expected_keys]), True)
        self.assertEqual(
            all([key in expected_keys for key in holiday_functions.keys()]), True)


if __name__ == "__main__":
    unittest.main()
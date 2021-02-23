# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

import unittest

from stf.feature_engineering.feature_free_days import create_holiday_functions

from test.utils import BaseTestCase

expected_keys = ['IsFeestdag', 'IsVoorjaarsvakantieZuid', 'IsBouwvakMidden',
                 'IsHerfstvakantieZuid', 'IsKoningsdag', 'IsZomervakantieMidden',
                 'IsHerfstvakantieMidden', 'IsBevrijdingsdag', 'IsPinksteren',
                 'IsBevrijdingsdag(brugdagen)', 'IsHerfstvakantieNoord',
                 'IsPasen', 'IsPaasdag', 'IsKerstvakantie', 'IsBouwvakNoord',
                 'IsVoorjaarsvakantieNoord', 'IsBouwvakZuid', 'IsHemelvaart(brugdagen)',
                 'IsMeivakantie', 'IsZomervakantieNoord', 'IsNieuwjaarsdag',
                 'IsVoorjaarsvakantieMidden', 'IsZomervakantieZuid', 'IsKerst',
                 'IsHemelvaart']


class GeneralTest(BaseTestCase):

    def test_create_holiday_functions(self):

        holiday_functions = create_holiday_functions()

        # Assert for every holiday a function is available and no extra functions are generated
        self.assertEqual(all([key in holiday_functions.keys()
                              for key in expected_keys]), True)
        self.assertEqual(
            all([key in expected_keys for key in holiday_functions.keys()]), True)


if __name__ == "__main__":
    unittest.main()
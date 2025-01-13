# SPDX-FileCopyrightText: 2017-2023 Contributors to the OpenSTEF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
import unittest
from test.unit.utils.base import BaseTestCase

from openstef.feature_engineering.holiday_features import (
    generate_holiday_feature_functions,
)

expected_keys = [
    "is_national_holiday",
    "is_nieuwjaarsdag",
    "is_goede_vrijdag",
    "is_eerste_paasdag",
    "is_tweede_paasdag",
    "is_koningsdag",
    "is_hemelvaart",
    "is_eerste_pinksterdag",
    "is_tweede_pinksterdag",
    "is_eerste_kerstdag",
    "is_tweede_kerstdag",
    "is_bridgeday",
    "is_schoolholiday",
    "is_voorjaarsvakantiezuid",
    "is_bouwvakmidden",
    "is_bouwvakzuid",
    "is_meivakantie",
    "is_zomervakantienoord",
    "is_herfstvakantienoord",
    "is_kerstvakantie",
    "is_voorjaarsvakantienoord",
    "is_voorjaarsvakantiemidden",
    "is_herfstvakantiemidden",
    "is_bouwvaknoord",
    "is_herfstvakantiezuid",
    "is_zomervakantiezuid",
    "is_zomervakantiemidden",
]


class GeneralTest(BaseTestCase):
    def test_create_holiday_functions(self):
        holiday_functions = generate_holiday_feature_functions(
            country_code="NL", years=[2023]
        )

        # Assert for every holiday a function is available and no extra functions are generated
        self.assertTrue(all([key in holiday_functions.keys() for key in expected_keys]))
        self.assertTrue(all([key in expected_keys for key in holiday_functions.keys()]))


if __name__ == "__main__":
    unittest.main()

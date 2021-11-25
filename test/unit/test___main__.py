# SPDX-FileCopyrightText: 2017-2021 Contributors to the OpenSTF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

import unittest
from test.utils import BaseTestCase
from openstf.__main__ import validate_task_name


class Test___Main__(BaseTestCase):
    """Test functionality of the __main__.py file"""

    def test_validate_task_name_happy(self):
        """Test using an existing task name should return None"""
        self.assertIsNone(validate_task_name("create_forecast"))


if __name__ == "__main__":
    unittest.main()

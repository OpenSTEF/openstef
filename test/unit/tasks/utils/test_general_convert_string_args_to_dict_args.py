# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

import unittest

from stf.tasks.utils.general import convert_string_args_to_dict_args

from test.utils import BaseTestCase


class TestConvertStringArgsToDictArgs(BaseTestCase):
    def test_only_args(self):
        argstring = "3, text1"
        res = convert_string_args_to_dict_args(argstring)

        expected_result = dict(args=[3, "text1"], kwargs={})
        self.assertDictEqual(res, expected_result)
        return

    def test_convert_float(self):
        argstring = "3.4, text1.5"
        res = convert_string_args_to_dict_args(argstring)

        expected_result = dict(args=[3.4, "text1.5"], kwargs={})
        self.assertDictEqual(res, expected_result)
        return

    def test_only_kwargs(self):
        argstring = "opt1 = val1, opt2=5"
        res = convert_string_args_to_dict_args(argstring)

        expected_result = dict(args=[], kwargs=dict(opt1="val1", opt2=5))
        self.assertDictEqual(res, expected_result)
        return

    def test_args_and_kwargs(self):
        argstring = "3, text1, opt1 = val1, opt2=5"
        res = convert_string_args_to_dict_args(argstring)

        expected_result = dict(args=[3, "text1"], kwargs=dict(opt1="val1", opt2=5))
        self.assertDictEqual(res, expected_result)
        return


if __name__ == "__main__":
    unittest.main()

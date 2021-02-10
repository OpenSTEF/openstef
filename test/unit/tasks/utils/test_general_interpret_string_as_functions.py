# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

import unittest

from stf.tasks.utils.general import interpret_string_as_functions

from test.utils import BaseTestCase


class TestInterpretStringAsFunctions(BaseTestCase):
    def test_single_float_arg(self):
        res = interpret_string_as_functions("f", [3.812], {})
        self.assertEqual(res, "f(3.812, db)")
        return

    def test_single_str_arg(self):
        res = interpret_string_as_functions("f", ["text1"], {})
        self.assertEqual(res, "f('text1', db)")
        return

    def test_only_args(self):
        res = interpret_string_as_functions("f", [3, "text1"], {})
        self.assertEqual(res, "f(3, 'text1', db)")
        return

    def test_only_args_with_float(self):
        res = interpret_string_as_functions("f", [3.812, "text1"], {})
        self.assertEqual(res, "f(3.812, 'text1', db)")
        return

    def test_only_single_kwarg(self):
        res = interpret_string_as_functions("f", [], {"key1": "val1"})
        self.assertEqual(res, "f(db, key1='val1')")
        return

    def test_only_kwargs(self):
        res = interpret_string_as_functions("f", [], {"key1": "val1", "key2": 8.123})
        self.assertEqual(res, "f(db, key1='val1', key2=8.123)")
        return

    def test_complete_case(self):
        res = interpret_string_as_functions(
            "f", [4, 5.1, "text1"], {"key1": "val1", "key2": 8.123, "key3": 4}
        )
        self.assertEqual(res, "f(4, 5.1, 'text1', db, key1='val1', key2=8.123, key3=4)")
        return


if __name__ == "__main__":
    unittest.main()

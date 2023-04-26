# SPDX-FileCopyrightText: 2017-2022 Contributors to the OpenSTEF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
import unittest

from openstef.metrics.metrics import get_eval_metric_function, mae


class TestEvalMetricFunction(unittest.TestCase):
    def test_eval_metric(self):
        self.assertEqual(get_eval_metric_function("mae"), mae)  # add assertion here

    def test_eval_metric_exception(self):
        with self.assertRaises(KeyError):
            get_eval_metric_function("non-existing")

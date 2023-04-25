# SPDX-FileCopyrightText: 2017-2023 Contributors to the OpenSTEF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

import unittest
from test.unit.utils.data import TestData


class TestModelSpecifications(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.pj, self.modelspecs = TestData.get_prediction_job_and_modelspecs(307)

    def test_model_specifications_get(self):
        fn = self.modelspecs.__getitem__("feature_names")
        self.assertIsInstance(fn, list)

    def test_model_specifications_set(self):
        hyper_params_test = {"set": "succes"}
        self.modelspecs.__setitem__("hyper_params", hyper_params_test)
        self.assertEqual(hyper_params_test, self.modelspecs.hyper_params)

    def test_model_specifications_set_unknown_attribute(self):
        with self.assertRaises(AttributeError):
            self.modelspecs.__setitem__("unknown", "shouldn't set")


if __name__ == "__main__":
    unittest.main()

# SPDX-FileCopyrightText: 2017-2023 Contributors to the OpenSTEF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
from test.unit.utils.base import BaseTestCase

import unittest
import pandas as pd
import numpy as np
from sklearn.exceptions import NotFittedError
from openstef.feature_engineering.missing_values_transformer import (
    MissingValuesTransformer,
)


class MissingValuesTransformerTests(BaseTestCase):
    def setUp(self):
        self.data = pd.DataFrame(
            {"A": [np.nan, 2, 3], "B": [1, np.nan, 3], "C": [4, 5, np.nan], "D": [np.nan, np.nan, np.nan]}
        )

    def test_imputation_with_mean_strategy_fills_missing_values(self):
        transformer = MissingValuesTransformer(imputation_strategy="mean")
        transformed = transformer.fit_transform(self.data)
        self.assertEqual(transformed.isnull().sum().sum(), 0)
        self.assertAlmostEqual(transformed.loc[0, "A"], 2.5)
        self.assertAlmostEqual(transformed.loc[1, "B"], 2)

    def test_imputation_with_constant_strategy_fills_missing_values(self):
        transformer = MissingValuesTransformer(
            imputation_strategy="constant", fill_value=0
        )
        transformed = transformer.fit_transform(self.data)
        self.assertEqual(transformed.isnull().sum().sum(), 0)
        self.assertEqual(transformed.loc[0, "A"], 0)
        self.assertEqual(transformed.loc[1, "B"], 0)

    def test_columns_always_null_are_removed(self):
        transformer = MissingValuesTransformer()
        transformer.fit(self.data)
        self.assertNotIn("D", transformer.non_null_feature_names)

    def test_columns_with_missing_values_at_end_are_removed(self):
        transformer = MissingValuesTransformer()
        transformer.fit(self.data)
        self.assertNotIn("C", transformer.non_null_feature_names)

    def test_non_dataframe_input_is_converted_and_processed(self):
        transformer = MissingValuesTransformer(imputation_strategy="mean")
        array = np.array([[1, np.nan, np.nan], [np.nan, 2, np.nan]])
        transformed = transformer.fit_transform(array)
        self.assertIsInstance(transformed, pd.DataFrame)
        self.assertEqual(transformed.isnull().sum().sum(), 0)

    def test_fitting_transformer_without_strategy_keeps_data_unchanged(self):
        transformer = MissingValuesTransformer()
        transformed = transformer.fit_transform(self.data)
        pd.testing.assert_frame_equal(transformed, self.data.drop(columns=["C", "D"]))

    def test_calling_transform_before_fit_raises_error(self):
        transformer = MissingValuesTransformer()
        with self.assertRaises(NotFittedError):
            transformer.transform(self.data)

    def test_imputation_with_unsupported_strategy_raises_value_error(self):
        with self.assertRaises(ValueError):
            MissingValuesTransformer(imputation_strategy="unsupported")

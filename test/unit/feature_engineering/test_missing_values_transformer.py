# SPDX-FileCopyrightText: 2017-2023 Contributors to the OpenSTEF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
from test.unit.utils.base import BaseTestCase

import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError

from openstef.feature_engineering.missing_values_transformer import (
    MissingValuesTransformer,
)


class MissingValuesTransformerTests(BaseTestCase):
    def setUp(self):
        self.data = pd.DataFrame(
            {
                "A": [np.nan, 2, 3, 4],
                "B": [3, np.nan, 4, 5],
                "C": [3, 4, 5, np.nan],
                "D": [np.nan, np.nan, np.nan, np.nan],
            },
            index=[0, 1, 1, 2],
        )

    def test_imputation_with_mean_strategy_fills_missing_values(self):
        transformer = MissingValuesTransformer(imputation_strategy="mean")
        transformed, _ = transformer.fit_transform(self.data)
        self.assertEqual(transformed.isnull().sum().sum(), 0)
        self.assertAlmostEqual(transformed.iloc[0]["A"], 3)
        self.assertAlmostEqual(transformed.iloc[1]["B"], 4)

    def test_imputation_with_constant_strategy_fills_missing_values(self):
        transformer = MissingValuesTransformer(
            imputation_strategy="constant", fill_value=0
        )
        transformed, _ = transformer.fit_transform(self.data)
        self.assertEqual(transformed.isnull().sum().sum(), 0)
        self.assertEqual(transformed.iloc[0]["A"], 0)
        self.assertEqual(transformed.iloc[1]["B"], 0)

    def test_columns_always_null_are_removed(self):
        transformer = MissingValuesTransformer()
        transformer.fit(self.data)
        self.assertNotIn("D", transformer.non_null_feature_names)

    def test_determining_non_trailing_null_rows(self):
        transformer = MissingValuesTransformer(no_fill_future_values_features=["C"])
        transformer.fit(self.data)
        non_trailing_null_rows = transformer._determine_trailing_null_rows(
            self.data[transformer.non_null_feature_names]
        )
        pd.testing.assert_series_equal(
            non_trailing_null_rows,
            pd.Series([True, True, True, False], index=[0, 1, 1, 2]),
        )

    def test_fitting_with_labels_removes_rows_with_trailing_nulls(self):
        transformer = MissingValuesTransformer(no_fill_future_values_features=["C"])
        _, y_transformed = transformer.fit_transform(
            self.data, y=pd.Series([1, 2, 3, 4], index=self.data.index)
        )
        self.assertEqual(y_transformed.tolist(), [1, 2, 3])

    def test_non_dataframe_input_is_converted_and_processed(self):
        transformer = MissingValuesTransformer(imputation_strategy="mean")
        array = np.array([[1, np.nan, np.nan], [np.nan, 2, np.nan]])
        transformed, _ = transformer.fit_transform(array)
        self.assertIsInstance(transformed, pd.DataFrame)
        self.assertEqual(transformed.isnull().sum().sum(), 0)
        self.assertEqual(transformed.shape, (2, 2))

    def test_fitting_transformer_without_strategy_keeps_valid_data_unchanged(self):
        transformer = MissingValuesTransformer()
        transformed, _ = transformer.fit_transform(self.data)
        pd.testing.assert_frame_equal(transformed, self.data.drop(columns=["D"]))

    def test_call_transform_on_fitted_transformer_does_not_remove_trailing_null_rows(
        self,
    ):
        transformer = MissingValuesTransformer(no_fill_future_values_features=["C"])
        transformer.fit(self.data)
        new_data = pd.DataFrame(
            {
                "A": [1, 2, 3, 4],
                "B": [1, 2, 3, 4],
                "C": [1, 2, 3, 4],
                "D": [1, 2, 3, 4],
            },
            index=[0, 1, 1, 2],
        )
        transformed = transformer.transform(new_data)
        pd.testing.assert_frame_equal(transformed, new_data.drop(columns=["D"]))

    def test_calling_transform_before_fit_raises_error(self):
        transformer = MissingValuesTransformer()
        with self.assertRaises(NotFittedError):
            transformer.transform(self.data)

    def test_imputation_with_unsupported_strategy_raises_value_error(self):
        with self.assertRaises(ValueError):
            MissingValuesTransformer(imputation_strategy="unsupported")

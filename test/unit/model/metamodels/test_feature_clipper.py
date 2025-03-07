# SPDX-FileCopyrightText: 2017-2023 Contributors to the OpenSTEF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
import unittest

import pandas as pd

from openstef.model.metamodels.feature_clipper import FeatureClipper


class TestFeatureClipper(unittest.TestCase):
    def setUp(self):
        # Training and test data with float values
        self.df_train = pd.DataFrame(
            {"A": [1.0, 2.0, 3.0], "B": [10.0, 20.0, 30.0], "C": [100.0, 200.0, 300.0]}
        )
        self.df_test = pd.DataFrame(
            {"A": [0.5, 4.0], "B": [5.0, 35.0], "C": [150.0, 350.0]}
        )
        # Initialize FeatureClipper with columns to clip
        self.clipper = FeatureClipper(
            columns=["A", "B", "D"]
        )  # 'D' doesn't exist in data

    def test_fit(self):
        """Test if the fit method correctly computes min and max values."""
        self.clipper.fit(self.df_train)
        self.assertEqual(self.clipper.feature_ranges["A"], (1.0, 3.0))
        self.assertEqual(self.clipper.feature_ranges["B"], (10.0, 30.0))
        self.assertNotIn("D", self.clipper.feature_ranges)

    def test_transform(self):
        """Test if the transform method correctly clips values."""
        self.clipper.fit(self.df_train)
        transformed_df = self.clipper.transform(self.df_test)
        expected_df = pd.DataFrame(
            {
                "A": [1.0, 3.0],  # Clipped to range [1.0, 3.0]
                "B": [10.0, 30.0],  # Clipped to range [10.0, 30.0]
                "C": [150.0, 350.0],  # Unchanged
            }
        )
        pd.testing.assert_frame_equal(transformed_df, expected_df)

    def test_invalid_column(self):
        """Test behavior when a column that doesn't exist is specified."""
        clipper_with_invalid_column = FeatureClipper(columns=["E"])
        clipper_with_invalid_column.fit(self.df_train)
        self.assertNotIn("E", clipper_with_invalid_column.feature_ranges)

    def test_invalid_input_type(self):
        """Test behavior when input is not a DataFrame."""
        with self.assertRaises(ValueError):
            self.clipper.fit([1.0, 2.0, 3.0])  # Invalid input type

    def test_transform_without_fit(self):
        """Test behavior when transform is called without fitting."""
        clipper = FeatureClipper(columns=["A", "B"])
        transformed_df = clipper.transform(self.df_test)
        # The transform should return the original DataFrame unchanged
        pd.testing.assert_frame_equal(transformed_df, self.df_test)

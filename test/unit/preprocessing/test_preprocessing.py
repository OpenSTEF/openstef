# SPDX-FileCopyrightText: 2017-2023 Contributors to the OpenSTEF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

from test.unit.utils.base import BaseTestCase

import pandas as pd

from openstef.preprocessing import preprocessing


class TestPreprocessing(BaseTestCase):
    def test_replace_repeated_values_with_nan(self):
        # Arrange
        data_no_repeated_sequences = [0, 1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1]
        data_two_repeated_sequences = [0, 1, 2, 2, 2, 2, 3, 4, 2, 2, 2, 5]

        df = pd.DataFrame(
            {
                "Column1": data_no_repeated_sequences,
                "Column2": data_two_repeated_sequences,
            }
        )

        # Act
        df_no_repeated_sequences = preprocessing.replace_repeated_values_with_nan(
            df, 2, "Column1"
        )

        df_repeated_sequences_below_threshold = (
            preprocessing.replace_repeated_values_with_nan(df, 8, "Column2")
        )

        df_one_repeated_sequence = preprocessing.replace_repeated_values_with_nan(
            df, 4, "Column2"
        )

        df_two_repeated_sequences = preprocessing.replace_repeated_values_with_nan(
            df, 3, "Column2"
        )

        # Assert
        self.assertFalse(df_no_repeated_sequences.isna().values.any())
        self.assertFalse(df_repeated_sequences_below_threshold.isna().values.any())
        self.assertEqual(df_one_repeated_sequence["Column2"].isna().values.sum(), 4)
        self.assertEqual(df_two_repeated_sequences["Column2"].isna().values.sum(), 7)

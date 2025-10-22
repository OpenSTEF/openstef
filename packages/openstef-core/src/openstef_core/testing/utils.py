# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Testing utilities for comparing pandas objects.

Provides matcher classes for use in test assertions when comparing pandas
DataFrames and Series with equality semantics.
"""

from typing import override

import pandas as pd


class IsSamePandas:
    """Utility class to allow comparison of pandas DataFrames in assertion / calls."""

    def __init__(self, pandas_obj: pd.DataFrame | pd.Series):
        """Matcher to check if two DataFrames are equal."""
        self.pandas_obj = pandas_obj

    @override
    def __eq__(self, other: object) -> bool:
        return isinstance(other, type(self.pandas_obj)) and self.pandas_obj.equals(other)  # type: ignore

    @override
    def __hash__(self) -> int:
        return hash(self.pandas_obj)

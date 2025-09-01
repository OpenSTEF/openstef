# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Custom exceptions for OpenSTEF core functionality.

This module defines specific exception types used throughout the OpenSTEF core
package to provide clear error reporting and handling for common failure cases.
"""


class MissingColumnsError(Exception):
    """Exception raised when required columns are missing from a DataFrame."""

    def __init__(self, missing_columns: list[str]):
        """Initialize the exception with the list of missing columns.

        Args:
            missing_columns: List of column names that are missing from the DataFrame.
        """
        self.missing_columns = missing_columns
        super().__init__(f"Missing required columns: {', '.join(missing_columns)}")


class InvalidColumnTypeError(Exception):
    """Exception raised when a DataFrame column has an invalid type."""

    def __init__(self, column: str, expected_type: str, actual_type: str):
        """Initialize the exception with details about the type mismatch.

        Args:
            column: Name of the column with the invalid type.
            expected_type: The expected data type for the column.
            actual_type: The actual data type found in the column.
        """
        message = f"Invalid type for column '{column}': expected {expected_type}, but got {actual_type}."
        super().__init__(message)


class TimeSeriesValidationError(Exception):
    """Exception raised for validation errors in time series datasets."""

    def __init__(self, message: str):
        """Initialize the exception with a descriptive error message.

        Args:
            message: Human-readable description of the validation error.
        """
        super().__init__(message)


class FlatlinerDetectedError(Exception):
    """Exception raised when a flatliner is detected in the data."""

    def __init__(self, message: str):
        """Initialize the exception with a descriptive error message.

        Args:
            message: Human-readable description of the flatliner detection error.
        """
        super().__init__(message)


__all__ = ["FlatlinerDetectedError", "MissingColumnsError", "TimeSeriesValidationError"]

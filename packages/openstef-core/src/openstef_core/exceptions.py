# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0


class MissingColumnsError(Exception):
    """Exception raised when required columns are missing from a DataFrame."""

    def __init__(self, missing_columns: list[str]):
        self.missing_columns = missing_columns
        super().__init__(f"Missing required columns: {', '.join(missing_columns)}")


class TimeSeriesValidationError(Exception):
    """Exception raised for validation errors in time series datasets."""

    def __init__(self, message: str):
        super().__init__(message)


__all__ = ["MissingColumnsError", "TimeSeriesValidationError"]

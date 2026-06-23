# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Custom exceptions for OpenSTEF core functionality.

This module defines specific exception types used throughout the OpenSTEF core
package to provide clear error reporting and handling for common failure cases.
"""

from collections.abc import Sequence


class MissingExtraError(Exception):
    """Exception raised when an optional dependency for a feature is not installed."""

    def __init__(self, extra: str, package: str = "openstef-beam", *, install_extra: str | None = None):
        """Initialize the exception with the missing dependency and how to install it.

        Args:
            extra: Import/distribution name that failed to import (e.g. ``"onnxruntime"``).
            package: OpenSTEF package that owns the optional feature.
            install_extra: The extra group to install for this feature (e.g. ``"cpu"``),
                yielding ``pip install {package}[{install_extra}]``. When ``None`` the hint
                installs ``extra`` directly. Prefer naming the extra group: not every package
                defines an ``[all]`` extra, and some groups are mutually exclusive.
        """
        self.extra = extra
        target = f"{package}[{install_extra}]" if install_extra else extra
        super().__init__(
            f"Optional dependency '{extra}' is required for this feature but is not installed. "
            f"Install it with `pip install {target}`."
        )


class MissingColumnsError(ValueError):
    """Exception raised when required columns are missing from a DataFrame."""

    def __init__(self, missing_columns: Sequence[str], columns: Sequence[str] | None = None):
        """Initialize the exception with the list of missing columns.

        Args:
            missing_columns: List of column names that are missing from the DataFrame.
            columns: Optional list of available column names in the DataFrame.
        """
        self.missing_columns = missing_columns
        self.columns = columns
        if columns is not None:
            message = (
                f"Missing required columns: {', '.join(missing_columns)}. Available columns: {', '.join(columns)}."
            )
        else:
            message = f"Missing required columns: {', '.join(missing_columns)}."
        super().__init__(message)


class InvalidColumnTypeError(TypeError):
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


class TimeSeriesValidationError(ValueError):
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


class ModelUnderperformingError(Exception):
    """Exception raised when a model is underperforming based on defined criteria."""

    def __init__(self, metric_name: str, metric_value: float, threshold: float):
        """Initialize the exception with a descriptive error message.

        Args:
            metric_name: Name of the performance metric that is underperforming.
            metric_value: The actual value of the performance metric.
            threshold: The defined threshold for acceptable performance.
        """
        message = (
            f"Model is underperforming: {metric_name} = {metric_value:.4f} "
            f"does not meet the threshold of {threshold:.4f}."
        )
        super().__init__(message)


class InsufficientlyCompleteError(Exception):
    """Exception raised when a dataset is not sufficiently complete."""

    def __init__(self, message: str):
        """Initialize the exception with a descriptive error message.

        Args:
            message: Human-readable description of the completeness error.
        """
        super().__init__(message)


class PredictError(Exception):
    """Exception raised for errors during forecasting operations."""


class InputValidationError(ValueError):
    """Exception raised for input validation errors."""


class ModelLoadingError(Exception):
    """Exception raised when a model fails to load properly."""


class NotFittedError(Exception):
    """Exception raised when a model or a transform is used before being fitted."""

    def __init__(self, component_class: str):
        """Initialize the exception with the name of the component class.

        Args:
            component_class: Name of the component class that was not fitted.
        """
        message = f"The {component_class} has not been fitted yet. Please call 'fit' before using it."
        super().__init__(message)


class SkipFitting(Exception):  # noqa: N818 - ignore, cause this is not an error
    """Exception raised to indicate that fitting should be skipped.

    This is used in scenarios where a model is determined to be recent enough
    or otherwise does not require re-fitting.
    """

    def __init__(self, reason: str):
        """Initialize the exception with an optional reason.

        Args:
            reason: Human-readable description of why fitting is being skipped.
        """
        super().__init__(reason)


class ModelNotFoundError(Exception):
    """Exception raised when a model is not found in storage."""

    def __init__(self, model_id: str):
        """Initialize the exception with the model identifier.

        Args:
            model_id: Identifier of the model that was not found.
        """
        message = f"The model with ID '{model_id}' was not found in storage."
        super().__init__(message)


class UnreachableStateError(Exception):
    """Exception raised when a code path that should be unreachable is executed.

    This indicates a violation of invariants or preconditions that should have
    been guaranteed by the system design. Typically used when configuration
    validation should have prevented reaching this state.
    """

    def __init__(self, message: str):
        """Initialize the exception with a descriptive error message.

        Args:
            message: Human-readable description of the unreachable state condition.
        """
        super().__init__(message)


class ConfigurationError(Exception):
    """Exception raised for errors in configuration settings."""

    def __init__(self, message: str):
        """Initialize the exception with a descriptive error message.

        Args:
            message: Human-readable description of the configuration error.
        """
        super().__init__(message)


__all__ = [
    "ConfigurationError",
    "FlatlinerDetectedError",
    "InsufficientlyCompleteError",
    "InvalidColumnTypeError",
    "MissingColumnsError",
    "MissingExtraError",
    "ModelLoadingError",
    "ModelNotFoundError",
    "NotFittedError",
    "PredictError",
    "TimeSeriesValidationError",
    "UnreachableStateError",
]

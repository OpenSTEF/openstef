# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Runtime invariant checking utilities.

Provides helper functions for asserting runtime invariants and contracts,
particularly for checking nullable values and other preconditions.
"""


def not_none[T](value: T | None) -> T:
    """Assert that a value is not None.

    Args:
        value: The value to check.

    Returns:
        The value if it is not None.

    Raises:
        ValueError: If the value is None.
    """
    if value is None:
        raise ValueError("Expected value to be not None")
    return value


__all__ = [
    "not_none",
]

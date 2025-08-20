# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Custom exceptions for the OpenSTEF BEAM package."""


class MissingExtraError(Exception):
    """Exception raised when an extra is missing in the extras list."""

    def __init__(self, extra: str):
        self.extra = extra
        super().__init__(
            f"The extras for {extra}. Please install it to use this module using `pip install stef-beam[{extra}]`."
        )

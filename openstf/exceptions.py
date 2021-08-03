# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

"""Define custom exception classes.
These are not complete, but will be added on a case by case basis
"""

from typing import Union

# Define custom exception
class NoPredictionException(Exception):
    """Custom exception if no historic predictions are available
    Attributes:
        pid -- prediction job id for which the exception occurred
        message -- explanation of the error
    """

    def __init__(
        self,
        pid: Union[int, str],
        message: str = "No historic predictions found." "Could not calc KPI",
    ):
        self.pid = pid
        self.message = message
        super().__init__(self.message)


class NoLoadException(Exception):
    """Custom Exception if no historic load is available
    Attributes:
        pid -- prediction job id for which the exception occurred
        message -- explanation of the error
    """

    def __init__(
        self,
        pid: Union[int, str],
        message: str = "No historic load found." "Could not calc KPI",
    ):
        self.pid = pid
        self.message = message
        super().__init__(self.message)

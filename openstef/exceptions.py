# SPDX-FileCopyrightText: 2017-2022 Contributors to the OpenSTEF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

"""openstef custom exceptions
"""
from datetime import datetime


# Define custom exception
class NoPredictedLoadError(Exception):
    """No predicted load for given datatime range"""

    def __init__(
        self,
        pid: int,
        message: str = "No predicted load found",
    ) -> Exception:
        self.pid = pid
        self.message = message
        super().__init__(self.message)


class NoRealisedLoadError(Exception):
    """No realised load for given datetime range"""

    def __init__(
        self,
        pid: int,
        message: str = "No realised load found",
    ) -> Exception:
        self.pid = pid
        self.message = message
        super().__init__(self.message)


class InputDataInvalidError(Exception):
    """Invalid input data"""


class InputDataInsufficientError(InputDataInvalidError):
    """Insufficient input data"""


class InputDataWrongColumnOrderError(InputDataInvalidError):
    """Wrong column order input data"""


class OldModelHigherScoreError(Exception):
    """Old model has a higher score then new model"""


class ModelWithoutStDev(Exception):
    """A machine learning model should have a valid standard deviation"""

# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

"""openstf custom exceptions
"""


# Define custom exception
class NoPredictedLoadError(Exception):
    """No predicted load for given datatime range"""


class NoRealisedLoadError(Exception):
    """No realised load for given datetime range"""


class InputDataInvalidError(Exception):
    """Invalid input data"""


class InputDataInsufficientError(InputDataInvalidError):
    """Insufficient input data"""


class InputDataWrongColumnOrderError(InputDataInvalidError):
    """Wrong column order input data"""


class OldModelHigherScoreError(Exception):
    """Old model has a higher score then new model"""

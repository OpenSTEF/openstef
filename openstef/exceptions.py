# SPDX-FileCopyrightText: 2017-2023 Contributors to the OpenSTEF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

"""Openstef custom exceptions."""


# Define custom exception
class NoPredictedLoadError(Exception):
    """No predicted load for given datatime range."""

    def __init__(
        self,
        pid: int,
        message: str = "No predicted load found",
    ):
        self.pid = pid
        self.message = message
        super().__init__(self.message)


class NoRealisedLoadError(Exception):
    """No realised load for given datetime range."""

    def __init__(
        self,
        pid: int,
        message: str = "No realised load found",
    ):
        self.pid = pid
        self.message = message
        super().__init__(self.message)


class InputDataInvalidError(Exception):
    """Invalid input data."""


class InputDataInsufficientError(InputDataInvalidError):
    """Insufficient input data."""


class InputDataWrongColumnOrderError(InputDataInvalidError):
    """Wrong column order input data."""


class InputDataOngoingZeroFlatlinerError(InputDataInvalidError):
    """All recent load measurements are zero."""


class OldModelHigherScoreError(Exception):
    """Old model has a higher score then new model."""


class ModelWithoutStDev(Exception):
    """A machine learning model should have a valid standard deviation."""


class ComponentForecastTooShortHorizonError(Exception):
    """Component forecasts should be available for at least 30 hours in advance."""


class PredictionJobException(Exception):
    """One or more prediction jobs raised an exception."""

    def __init__(self, metrics=None):
        super().__init__("One or more prediction jobs raised an exception.")
        if metrics is None:
            metrics = {}
        self.metrics = metrics


class SkipSaveTrainingForecasts(Exception):
    """If old model is better or too young, you don't need to save the traing forcast."""

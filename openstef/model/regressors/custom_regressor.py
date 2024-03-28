# SPDX-FileCopyrightText: 2017-2023 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
"""This module defines the custom regressor."""
import inspect
from abc import abstractmethod
from importlib import import_module
from typing import Type

import pandas as pd

from openstef.model.objective import (
    EVAL_METRIC,
    TEST_FRACTION,
    VALIDATION_FRACTION,
    RegressorObjective,
)
from openstef.model.regressors.regressor import OpenstfRegressor


class CustomOpenstfRegressor(OpenstfRegressor):
    """A custom regressor allows to load any custom model that is not included with openSTEF."""

    @staticmethod
    @abstractmethod
    def valid_kwargs() -> list[str]:
        ...

    @staticmethod
    @abstractmethod
    def objective() -> Type[RegressorObjective]:
        ...


def load_custom_model(custom_model_path) -> CustomOpenstfRegressor:
    """Load the external custom model."""
    path_elements = custom_model_path.split(".")
    module_path = ".".join(path_elements[:-1])
    module = import_module(module_path)
    model_name = path_elements[-1]
    model_class = getattr(module, model_name)

    if (
        not inspect.isclass(model_class)
        or inspect.isabstract(model_class)
        or not issubclass(model_class, CustomOpenstfRegressor)
    ):
        raise ValueError(
            f"The path {custom_model_path!r} does not correspond to a concrete"
            " CustomOpenstfRegressor subclass"
        )

    return model_class


def is_custom_type(model_type):
    return isinstance(model_type, str) and "." in model_type


def create_custom_objective(
    custom_model_path,
):
    model_class = load_custom_model(custom_model_path)
    return model_class.objective()

# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

from typing import List, Type
from abc import abstractmethod
from importlib import import_module
import pandas as pd
import inspect
from openstef.model.regressors.regressor import OpenstfRegressor
from openstef.model.objective import (
    RegressorObjective,
    TEST_FRACTION,
    VALIDATION_FRACTION,
    EVAL_METRIC,
)


class CustomOpenstfRegressor(OpenstfRegressor):
    @staticmethod
    @abstractmethod
    def valid_kwargs() -> List[str]:
        ...

    @property
    @abstractmethod
    def objective(self) -> Type[RegressorObjective]:
        ...


def load_custom_model(custom_model_path) -> CustomOpenstfRegressor:
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
            f"The path {custom_model_path!r} does not correspond to a concrete CustomOpenstfRegressor subclass"
        )

    return model_class


def is_custom_type(model_type):
    return isinstance(model_type, str) and "." in model_type


def create_custom_objective(
    model: CustomOpenstfRegressor,
    input_data: pd.DataFrame,
    test_fraction=TEST_FRACTION,
    validation_fraction=VALIDATION_FRACTION,
    eval_metric=EVAL_METRIC,
    verbose=False,
):
    return model.objective(
        model,
        input_data=input_data,
        test_fraction=test_fraction,
        validation_fraction=validation_fraction,
        eval_metric=eval_metric,
        verbose=verbose,
    )

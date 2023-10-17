# SPDX-FileCopyrightText: 2017-2023 Contributors to the OpenSTEF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
"""Specifies the split function dataclass."""
import inspect
import json
from importlib import import_module
from typing import Any, Sequence, Union, TypeVar

from pydantic.v1 import BaseModel

DataPrepClass = TypeVar("DataPrepClass")


class DataPrepDataClass(BaseModel):
    """Class that allows to specify a custom class to prepare the data (feature engineering , etc ...)."""

    klass: Union[str, type[DataPrepClass]]
    arguments: Union[
        str, dict[str, Any]
    ]  # JSON string holding the function parameters or dict

    def __getitem__(self, key: str):
        """Allows us to use subscription to get the items from the object."""
        return getattr(self, key)

    def __setitem__(self, key: str, value: any):
        """Allows us to use subscription to set the items in the object."""
        if hasattr(self, key):
            self.__dict__[key] = value
        else:
            raise AttributeError(f"{key} not an attribute of prediction job.")

    def _load_klass(
        self, required_arguments: Sequence[str] = None
    ) -> type[DataPrepClass]:
        """Load data prep class from path.

        Args:
            klass_path (str): The path to the data prep class

        Returns:
            klass (type[AbstractDataPreparation]): The loaded data prep class

        """
        if isinstance(self.klass, str):
            path_elements = self.klass.split(".")
            module_path = ".".join(path_elements[:-1])
            module = import_module(module_path)
            klass_name = path_elements[-1]
            klass = getattr(module, klass_name)
        else:
            klass = self.klass

        # Check that the klass accepts mandatory arguments
        if not inspect.isclass(klass):
            raise ValueError("The loaded object is not a class: {klass!r}")

        if required_arguments is not None:
            klass_params = set(inspect.signature(klass).parameters)

            if len(set(required_arguments) - klass_params) > 0:
                raise ValueError(
                    "The loaded data prep class does not have the required arguments"
                )

        return klass

    def _load_arguments(self) -> dict[str, Any]:
        """Load the arguments.

        Convert the arguments from JSON if they are given as strings or simply return them otherwise.

        Returns:
            arguments (dict[str, Any]): The additional arguments to be passed to the class

        """
        if isinstance(self.arguments, str):
            return json.loads(self.arguments)
        else:
            return self.arguments

    def load(
        self, required_arguments: Sequence[str] = None
    ) -> tuple[type[DataPrepClass], dict[str, Any]]:
        """Load the function and its arguments.

        If the function and the arguments are given as strings in the instane attributes, load them as Python objects
        otherwise just return them from the instance attributes.

        Args:
            required_arguments (list[str]): list of arguments the loaded class must have

        Returns:
            - class (type[AbstractDataPreparation])
            - arguments (dict[str, Any])

        """
        return self._load_klass(required_arguments), self._load_arguments()

# SPDX-FileCopyrightText: 2017-2022 Contributors to the OpenSTEF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

import inspect
import json
from importlib import import_module
from typing import Any, Callable, Dict, Sequence, Union

from pydantic import BaseModel


class SplitFuncDataClass(BaseModel):
    function: Union[str, Callable]
    arguments: Union[
        str, Dict[str, Any]
    ]  # JSON string holding the function parameters or dict

    def __getitem__(self, key: str):
        """Allows us to use subscription to get the items from the object"""
        return getattr(self, key)

    def __setitem__(self, key: str, value: any):
        """Allows us to use subscription to set the items in the object"""
        if hasattr(self, key):
            self.__dict__[key] = value
        else:
            raise AttributeError(f"{key} not an attribute of prediction job.")

    def _load_split_function(self, required_arguments=None) -> Callable:
        """Load split function from path

        Args:
            func_path (str): The path to the split function

        Returns:
            split_func (Callable): The loaded split function
        """
        if isinstance(self.function, str):
            path_elements = self.function.split(".")
            module_path = ".".join(path_elements[:-1])
            module = import_module(module_path)
            func_name = path_elements[-1]
            split_func = getattr(module, func_name)
        else:
            split_func = self.function

        # Check that the function accepts mandatory arguments
        if not callable(split_func):
            raise ValueError("The loaded object is not callable: {func_path!r}")

        if required_arguments is not None:
            func_params = set(inspect.signature(split_func).parameters)

            if len(set(required_arguments) - func_params) > 0:
                raise ValueError(
                    "The loaded split function does not have the required arguments"
                )

        return split_func

    def _load_arguments(self) -> Dict[str, Any]:
        """Load the arguments.

        Convert the arguments from JSON if they are given as strings or simply return them otherwise.

        Returns:
            arguments (Dict[str, Any]): The additional arguments to be passed to he function
        """
        if isinstance(self.arguments, str):
            return json.loads(self.arguments)
        else:
            return self.arguments

    def load(
        self, required_arguments: Sequence[str] = None
    ) -> (Callable, Dict[str, Any]):
        """Load the function and its arguments

        If the function and the arguments are given as strings in the instane attributes, load them as Python objects
        otherwise just return them from the instance attributes.

        Args:
            required_arguments (List[str]): list of arguments the loaded function must have

        Returns:
            - function (Callable)
            - arguments (Dict[str, Any])
        """
        return self._load_split_function(required_arguments), self._load_arguments()

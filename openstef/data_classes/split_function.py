# SPDX-FileCopyrightText: 2017-2022 Contributors to the OpenSTEF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

import json
from importlib import import_module
from typing import Union, Dict, Callable, Any

import inspect
from pydantic import BaseModel


class SplitFuncDataClass(BaseModel):
    function: Union[str, Callable]
    arguments: Union[str, Dict[str, Any]]  # JSON string holding the function parameters or dict

    def __getitem__(self, item):
        """Allows us to use subscription to get the items from the object"""
        return getattr(self, item)

    def __setitem__(self, key: str, value: any):
        """Allows us to use subscription to set the items in the object"""
        if hasattr(self, key):
            self.__dict__[key] = value
        else:
            raise AttributeError(f"{key} not an attribute of prediction job.")

    def _load_split_function(self, required_arguments=None):
        """ Load split function from path

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
                raise ValueError("The loaded split function does not have the required arguments")

            return split_func

    def _load_arguments(self):
        if isinstance(self.arguments, str):
            return json.loads(self.arguments)
        else:
            return self.arguments

    def load(self, required_arguments=None):
        return (
            self._load_split_function(required_arguments),
            self._load_arguments()
        )

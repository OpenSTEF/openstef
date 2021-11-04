from typing import List
from abc import abstractmethod
from importlib import import_module

from .regressor import OpenstfRegressor
from ..objective import RegressorObjective


class CustomOpenstfRegressor(OpenstfRegressor):
    @staticmethod
    @abstractmethod
    def valid_kwargs() -> List[str]:
        ...

    @property
    @abstractmethod
    def objective(self) -> RegressorObjective:
        ...


def create_custom_model(custom_model_path, **kwargs) -> CustomOpenstfRegressor:
    path_elts = custom_model_path.split(".")
    module_path=".".join(path_elts[:-1])
    module = import_module(module_path)
    model_name = path_elts[-1]
    model_class = getattr(module, model_name)

    if type(model_class) != type or not issubclass(model_class, CustomOpenstfRegressor):
        raise ValueError(f"The path {custom_model_path!r} does not correspond to a CustomOpenstfRegressor subclass")

    model_kwargs = {
        k: v
        for k, v in kwargs.items()
        if k in model_class.valid_kwargs()
    }

    return model_class(**model_kwargs)

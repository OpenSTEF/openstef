# SPDX-FileCopyrightText: 2017-2023 Contributors to the OpenSTEF project <korte.termijn.prognoses@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0
"""Specifies the dataclass for model specifications."""
from typing import Optional, Union

from pydantic.v1 import BaseModel


class ModelSpecificationDataClass(BaseModel):
    """Holds all information regarding the training procces of a specific model."""

    id: Union[int, str]
    hyper_params: Optional[dict] = {}
    """Hyperparameters that should be used during training."""
    feature_names: Optional[list] = None
    """Features that should be used during training."""
    feature_modules: Optional[list] = []
    """Feature modules that should be used during training."""

    def __getitem__(self, item: str) -> any:
        """Allows us to use subscription to get the items from the object."""
        return getattr(self, item)

    def __setitem__(self, key: str, value: any) -> None:
        """Allows us to use subscription to set the items in the object."""
        if hasattr(self, key):
            self.__dict__[key] = value
        else:
            raise AttributeError(f"{key} not an attribute of model specifications.")

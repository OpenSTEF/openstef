# SPDX-FileCopyrightText: 2017-2023 Contributors to the OpenSTEF project <korte.termijn.prognoses@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0
"""Specifies the dataclass for model specifications."""

from typing import Any, Optional, Union

from pydantic import BaseModel, Field


class ModelSpecificationDataClass(BaseModel):
    """Holds all information regarding the training procces of a specific model."""

    id: Union[int, str] = Field(description="The model id.")
    hyper_params: Optional[dict] = Field(
        default={}, description="Hyperparameters that should be used during training."
    )
    feature_names: Optional[list] = Field(
        default=None, description="Features that should be used during training."
    )
    feature_modules: Optional[list] = Field(
        default=[], description="Modules that should be used during training."
    )

    def __getitem__(self, item: str) -> Any:
        """Allows us to use subscription to get the items from the object."""
        return getattr(self, item)

    def __setitem__(self, key: str, value: Any) -> None:
        """Allows us to use subscription to set the items in the object."""
        if hasattr(self, key):
            self.__dict__[key] = value
        else:
            raise AttributeError(f"{key} not an attribute of model specifications.")

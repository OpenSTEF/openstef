# SPDX-FileCopyrightText: 2017-2022 Contributors to the OpenSTEF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

from typing import Union, Optional, List

from pydantic import BaseModel


class PredictionJobDataClass(BaseModel):
    id: Union[int, str]
    model: str
    forecast_type: str
    horizon_minutes: int
    resolution_minutes: int
    lat: float
    lon: float
    train_components: bool
    name: str
    description: Optional[str]
    quantiles: Optional[List[float]]

    def __getitem__(self, item):
        """Allows us to use subscription to get the items from the object"""
        return getattr(self, item)

    def __setitem__(self, key: str, value: any):
        """Allows us to use subscription to set the items in the object"""
        if hasattr(self, key):
            self.__dict__[key] = value
        else:
            raise AttributeError(f"{key} not an attribute of prediction job.")

# SPDX-FileCopyrightText: 2017-2023 Contributors to the OpenSTEF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
"""Specifies the prediction job dataclass."""
from typing import Optional, Union

from pydantic.v1 import BaseModel

from openstef.data_classes.data_prep import DataPrepDataClass
from openstef.data_classes.model_specifications import ModelSpecificationDataClass
from openstef.data_classes.split_function import SplitFuncDataClass
from openstef.enums import PipelineType, BiddingZone, AggregateFunction


class PredictionJobDataClass(BaseModel):
    """Holds all information about the specific forecast that has to be made."""

    id: Union[int, str]
    """The predictions job id (often abreviated as pid)."""
    model: str
    """The model type that should be used.

    Options are:
        - ``"xgb"``
        - ``"xgb_quantile"``
        - ``"lgb"``
        - ``"linear"``
        - ``"linear_quantile"``
        - ``"gblinear_quantile"``
        - ``"xgb_multioutput_quantile"``
        - ``"flatliner"``

    If unsure what to pick, choose ``"xgb"``.

    """
    model_kwargs: Optional[dict]
    """The model parameters that should be used."""
    forecast_type: str
    """The type of forecasts that should be made.

    Options are:
        - ``"demand"``
        - ``"wind"``
        - ``"basecase"``

    If unsure what to pick, choose ``"demand"``.

    """
    horizon_minutes: Optional[int] = 2880
    """The horizon of the desired forecast in minutes used in tasks. Defaults to 2880 minutes (i.e. 2 days)."""
    resolution_minutes: int
    """The resolution of the desired forecast in minutes."""
    lat: Optional[float] = 52.132633
    """Latitude of the forecasted location in degrees. Used for fetching weather data in tasks, calculating derrived features and component splitting."""
    lon: Optional[float] = 5.291266
    """Longitude of the forecasted location in degrees. Used for fetching weather data in tasks, calculating derrived features and component splitting."""
    name: str
    """Bidding zone is used to determine the electricity price. It is also used to determine the holidays that should be used. Currently only ENTSO-E bidding zones are supported."""
    electricity_bidding_zone: Optional[BiddingZone] = BiddingZone.NL
    """Name of the forecast, e.g. the location name."""
    train_components: Optional[bool]
    """Whether splitting the forecasts in wind, solar, rest is desired."""
    description: Optional[str]
    """Optional description of the prediction job for human reference."""
    quantiles: Optional[list[float]]
    """Quantiles that have to be forecasted."""
    train_split_func: Optional[SplitFuncDataClass]
    """Optional custom splitting function for operational procces."""
    backtest_split_func: Optional[SplitFuncDataClass]
    """Optional custom splitting function for backtesting."""
    train_horizons_minutes: Optional[list[int]]
    """List of horizons that should be taken into account during training."""
    default_modelspecs: Optional[ModelSpecificationDataClass]
    """Default model specifications"""
    save_train_forecasts: bool = False
    """Indicate wether the forecasts produced during the training process should be saved."""
    completeness_threshold: float = 0.5
    """Minimum fraction of data that should be available for making a regular forecast."""
    minimal_table_length: int = 100
    """Minimum length (in rows) of the forecast input for making a regular forecast."""
    flatliner_threshold_minutes: int = 1440
    """Number of minutes that the load has to be constant to detect a flatliner. """
    data_balancing_ratio: Optional[float] = None
    """If data balancing is enabled, the data will be balanced with data from 1 year 
    ago in the future."""
    rolling_aggregate_features: Optional[list[AggregateFunction]] = None
    """If not None, rolling aggregate(s) of load will be used as features in the model."""
    depends_on: Optional[list[Union[int, str]]]
    """Link to another prediction job on which this prediction job might depend."""
    sid: Optional[str]
    """Only required for create_solar_forecast task"""
    turbine_type: Optional[str]
    """Only required for create_wind_forecast task"""
    n_turbines: Optional[float]
    """Only required for create_wind_forecast task"""
    hub_height: Optional[float]
    """Only required for create_wind_forecast task"""
    pipelines_to_run: list[PipelineType] = [
        PipelineType.TRAIN,
        PipelineType.HYPER_PARMATERS,
        PipelineType.FORECAST,
    ]
    """The pipelines to run for this pj"""
    alternative_forecast_model_pid: Optional[Union[int, str]]
    """The pid that references another prediction job from which the model should be used for making forecasts."""
    data_prep_class: Optional[DataPrepDataClass]
    """The import string for the custom data prep class"""

    class Config:
        """Pydantic model configuration.

        This following configuration is needed to prevent ids in "depends_on" to be converted from int to str when we
        use integer ids.

        """

        smart_union = True

    def __getitem__(self, item: str) -> any:
        """Allows us to use subscription to get the items from the object."""
        return getattr(self, item)

    def __setitem__(self, key: str, value: any) -> None:
        """Allows us to use subscription to set the items in the object."""
        if hasattr(self, key):
            self.__dict__[key] = value
        else:
            raise AttributeError(f"{key} not an attribute of prediction job.")

    def get(self, key: str, default: any = None) -> any:
        """Allows to use the get functions similar to a python dict."""
        if hasattr(self, key):
            return getattr(self, key)
        else:
            return default

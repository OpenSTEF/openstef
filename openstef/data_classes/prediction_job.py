# SPDX-FileCopyrightText: 2017-2023 Contributors to the OpenSTEF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
"""Specifies the prediction job dataclass."""

from typing import Any, Optional, Union

from pydantic import BaseModel, Field

from openstef.data_classes.data_prep import DataPrepDataClass
from openstef.data_classes.model_specifications import ModelSpecificationDataClass
from openstef.data_classes.split_function import SplitFuncDataClass
from openstef.enums import AggregateFunction, BiddingZone, PipelineType


class PredictionJobDataClass(BaseModel):
    """Holds all information about the specific forecast that has to be made."""

    id: Union[int, str] = Field(
        ..., description="The predictions job id (often abreviated as pid)."
    )
    model: str = Field(
        ...,
        description="The model type that should be used. Options are: 'xgb', 'xgb_quantile', 'lgb', 'linear', 'linear_quantile', 'gblinear_quantile', 'xgb_multioutput_quantile', 'flatliner'.",
    )

    model_kwargs: Optional[dict] = Field(
        default=None, description="The model parameters that should be used."
    )

    forecast_type: str = Field(
        ...,
        description="The type of forecasts that should be made. Options are: 'demand', 'wind', 'basecase'. If unsure what to pick, choose 'demand'.",
    )
    horizon_minutes: Optional[int] = Field(
        2880,
        description="The horizon of the desired forecast in minutes used in tasks. Defaults to 2880 minutes (i.e. 2 days).",
    )

    resolution_minutes: int = Field(
        60, description="The resolution of the desired forecast in minutes."
    )
    lat: Optional[float] = Field(
        52.132633,
        description="Latitude of the forecasted location in degrees. Used for fetching weather data in tasks, calculating derrived features and component splitting.",
    )
    lon: Optional[float] = Field(
        5.291266,
        description="Longitude of the forecasted location in degrees. Used for fetching weather data in tasks, calculating derrived features and component splitting.",
    )
    name: str = Field(..., description="Name of the forecast, e.g. the location name.")

    electricity_bidding_zone: Optional[BiddingZone] = Field(
        BiddingZone.NL,
        description="The bidding zone of the forecasted location. Used for fetching electricity prices in tasks. It is also used to determine the holidays that should be used. Currently only ENTSO-E bidding zones are supported.",
    )
    train_components: Optional[bool] = Field(
        None,
        description="Whether splitting the forecasts in wind, solar, rest is desired.",
    )
    description: Optional[str] = Field(
        None,
        description="Optional description of the prediction job for human reference.",
    )

    quantiles: Optional[list[float]] = Field(
        None,
        description="Quantiles that have to be forecasted. Only used for quantile models.",
    )
    train_split_func: Optional[SplitFuncDataClass] = Field(
        None, description="Optional custom splitting function for operational procces."
    )
    backtest_split_func: Optional[SplitFuncDataClass] = Field(
        None, description="Optional custom splitting function for backtesting."
    )
    train_horizons_minutes: Optional[list[int]] = Field(
        None,
        description="List of horizons that should be taken into account during training.",
    )
    default_modelspecs: Optional[ModelSpecificationDataClass] = Field(
        None, description="Default model specifications"
    )
    save_train_forecasts: bool = Field(
        False,
        description="Indicate wether the forecasts produced during the training process should be saved.",
    )
    completeness_threshold: float = Field(
        0.5,
        description="Minimum fraction of data that should be available for making a regular forecast.",
    )
    minimal_table_length: int = Field(
        100,
        description="Minimum length (in rows) of the forecast input for making a regular forecast.",
    )
    flatliner_threshold_minutes: int = Field(
        1440,
        description="Number of minutes that the load has to be constant to detect a flatliner.",
    )
    detect_non_zero_flatliner: bool = Field(
        False,
        description="If True, flatliners are also detected on non-zero values (median of the load).",
    )
    data_balancing_ratio: Optional[float] = Field(
        None,
        description="If data balancing is enabled, the data will be balanced with data from 1 year ago in the future.",
    )
    rolling_aggregate_features: Optional[list[AggregateFunction]] = Field(
        [],
        description="If not None, rolling aggregate(s) of load will be used as features in the model.",
    )
    depends_on: Optional[list[Union[int, str]]] = Field(
        [],
        description="Link to another prediction job on which this prediction job might depend.",
    )
    sid: Optional[str] = Field(
        None, description="Only required for create_solar_forecast task"
    )
    turbine_type: Optional[str] = Field(
        None, description="Only required for create_wind_forecast task"
    )
    n_turbines: Optional[float] = Field(
        None, description="Only required for create_wind_forecast task"
    )
    hub_height: Optional[float] = Field(
        None, description="Only required for create_wind_forecast task"
    )
    pipelines_to_run: list[PipelineType] = Field(
        [PipelineType.TRAIN, PipelineType.HYPER_PARMATERS, PipelineType.FORECAST],
        description="The pipelines to run for this pj",
    )
    alternative_forecast_model_pid: Optional[Union[int, str]] = Field(
        None,
        description="The pid that references another prediction job from which the model should be used for making forecasts.",
    )
    data_prep_class: Optional[DataPrepDataClass] = Field(
        None, description="The import string for the custom data prep class"
    )

    def __getitem__(self, item: str) -> Any:
        """Allows us to use subscription to get the items from the object."""
        return getattr(self, item)

    def __setitem__(self, key: str, value: Any) -> None:
        """Allows us to use subscription to set the items in the object."""
        if hasattr(self, key):
            self.__dict__[key] = value
        else:
            raise AttributeError(f"{key} not an attribute of prediction job.")

    def get(self, key: str, default: Any = None) -> Any:
        """Allows to use the get functions similar to a python dict."""
        if hasattr(self, key):
            return getattr(self, key)
        else:
            return default

from abc import ABC, abstractmethod
from typing import Self, Type, cast, override

from openstef_core.base_model import BaseModel
from openstef_core.datasets.validated_datasets import ForecastDataset, ForecastInputDataset
from openstef_core.types import LeadTime
from openstef_models.models.forecasting.mixins import (
    ForecasterConfig,
    BaseForecaster,
    HorizonForecasterConfig,
    BaseHorizonForecaster,
    ModelState,
)


class MultiHorizonForecasterConfig[FC: ForecasterConfig](ForecasterConfig):
    forecaster_config: FC


class MultiHorizonForecasterState[FC: HorizonForecasterConfig](BaseModel):
    config: MultiHorizonForecasterConfig[FC]
    forecasters: dict[LeadTime, ModelState]


class MultiHorizonForecasterAdapter[
    FC: HorizonForecasterConfig,
    F: BaseHorizonForecaster,
](BaseForecaster, ABC):
    _config: MultiHorizonForecasterConfig[FC]
    _horizon_forecasters: dict[LeadTime, F]

    def __init__(
        self,
        config: MultiHorizonForecasterConfig[FC],
        horizon_forecasters: dict[LeadTime, F],
    ) -> None:
        self._config = config
        self._horizon_forecasters = horizon_forecasters

    @classmethod
    @abstractmethod
    def get_forecaster_type(cls) -> type[F]:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def create_forecaster(cls, config: FC) -> F:
        raise NotImplementedError

    @classmethod
    def create(
        cls,
        config: MultiHorizonForecasterConfig[FC],
    ) -> Self:
        return cls(
            config=config,
            horizon_forecasters={
                lead_time: cls.create_forecaster(config=config.forecaster_config.with_horizon(lead_time))
                for lead_time in config.horizons
            },
        )

    @override
    def get_state(self) -> ModelState:
        return MultiHorizonForecasterState[FC](
            config=self._config,
            forecasters={
                lead_time: forecaster.get_state() for lead_time, forecaster in self._horizon_forecasters.items()
            },
        )

    @classmethod
    @override
    def from_state(cls, state: ModelState) -> Self:
        state = cast(MultiHorizonForecasterState[FC], state)
        forecaster_type = cls.get_forecaster_type()

        return cls(
            config=state.config,
            horizon_forecasters={
                lead_time: forecaster_type.from_state(forecaster_state)
                for lead_time, forecaster_state in state.forecasters.items()
            },
        )

    @override
    def fit(self, input_data: dict[LeadTime, ForecastInputDataset]) -> None:
        for lead_time, forecaster in self._horizon_forecasters.items():
            forecaster.fit_horizon(input_data[lead_time])

    @override
    def predict(self, input_data: dict[LeadTime, ForecastInputDataset]) -> ForecastDataset:
        predictions = {
            lead_time: forecaster.predict_horizon(input_data[lead_time])
            for lead_time, forecaster in self._horizon_forecasters.items()
        }
        return combine_horizon_forecasts(predictions)


def combine_horizon_forecasts(forecasts: dict[LeadTime, ForecastDataset]) -> ForecastDataset:
    raise NotImplementedError

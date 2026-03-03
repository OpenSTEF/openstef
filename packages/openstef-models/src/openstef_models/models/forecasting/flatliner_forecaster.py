# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Simple constant zero forecasting model.

Provides basic forecasting model that predict constant flatliner zero values. It can be used
when a flatline (non-)zero measurement is observed in the past and expected in the future.
"""

from typing import override

import pandas as pd
from pydantic import Field, PrivateAttr

from openstef_core.datasets.validated_datasets import ForecastDataset, ForecastInputDataset, TimeSeriesDataset
from openstef_core.mixins.predictor import HyperParams
from openstef_models.explainability.mixins import ContributionsMixin, ExplainableForecaster
from openstef_models.models.forecasting.forecaster import Forecaster

MODEL_CODE_VERSION = 1


class FlatlinerForecaster(Forecaster, ExplainableForecaster, ContributionsMixin):
    """Flatliner forecaster that predicts a flatline of zeros or median.

    A simple forecasting model that always predicts zero (or the median of historical
    load measurements if configured) for all horizons and quantiles.

    Invariants:
        - Configuration quantiles determine the number of prediction outputs
        - Zeros (or median values) are predicted for all horizons and quantiles

    Example:
        >>> from openstef_core.types import LeadTime, Quantile
        >>> from datetime import timedelta
        >>> forecaster = FlatlinerForecaster(
        ...     quantiles=[Quantile(0.5), Quantile(0.1), Quantile(0.9)],
        ...     horizons=[LeadTime(timedelta(hours=1)), LeadTime(timedelta(hours=2))],
        ... )
        >>> forecaster.fit(training_data)  # doctest: +SKIP
        >>> predictions = forecaster.predict(test_data)  # doctest: +SKIP

    See Also:
        Forecaster: Base class for forecasting models that predict multiple horizons.
    """

    predict_median: bool = Field(
        default=False,
        description="If True, predict the median of load measurements instead of zero.",
    )

    hyperparams: HyperParams = Field(
        default_factory=HyperParams,
        description="Model hyperparameters (no tuning parameters for flatliner).",
    )

    _median_value: float | None = PrivateAttr(default=None)

    @property
    @override
    def hparams(self) -> HyperParams:
        return self.hyperparams

    @property
    @override
    def is_fitted(self) -> bool:
        if self.predict_median:
            return self._median_value is not None
        return True

    @override
    def fit(
        self,
        data: ForecastInputDataset,
        data_val: ForecastInputDataset | None = None,
    ) -> None:
        if self.predict_median:
            self._median_value = float(data.target_series.median())

    @override
    def predict(self, data: ForecastInputDataset) -> ForecastDataset:
        forecast_index = data.create_forecast_range(horizon=self.max_horizon)

        prediction_value = self._median_value if self.predict_median else 0.0

        return ForecastDataset(
            data=pd.DataFrame(
                data={quantile.format(): prediction_value for quantile in self.quantiles},
                index=forecast_index,
            ),
            sample_interval=data.sample_interval,
        )

    @property
    @override
    def feature_importances(self) -> pd.DataFrame:
        return pd.DataFrame(
            data=[0.0],
            index=["load"],
            columns=[quantile.format() for quantile in self.quantiles],
        )

    @override
    def predict_contributions(self, data: ForecastInputDataset) -> TimeSeriesDataset:
        """Return zero contributions since flatliner has no features."""
        input_data = data.input_data(start=data.forecast_start)
        contribs_df = pd.DataFrame(0.0, index=input_data.index, columns=["bias"])
        return TimeSeriesDataset(data=contribs_df, sample_interval=data.sample_interval)

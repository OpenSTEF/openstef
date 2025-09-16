# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""High-level forecasting model that orchestrates the complete prediction pipeline.

Combines feature engineering, forecasting, and postprocessing into a unified interface.
Handles both single-horizon and multi-horizon forecasters while providing consistent
data transformation and validation.
"""

from datetime import datetime

from openstef_core.datasets import TimeSeriesDataset, VersionedTimeSeriesDataset
from openstef_core.datasets.validated_datasets import ForecastDataset, ForecastInputDataset
from openstef_core.exceptions import ConfigurationError, NotFittedError, UnreachableStateError
from openstef_core.types import LeadTime
from openstef_models.models.forecasting import BaseForecaster, BaseHorizonForecaster
from openstef_models.transforms import FeaturePipeline, ForecastTransformPipeline


class ForecastingModel:
    """Complete forecasting pipeline combining preprocessing, prediction, and postprocessing.

    Orchestrates the full forecasting workflow by managing feature engineering,
    model training/prediction, and result postprocessing. Automatically handles
    the differences between single-horizon and multi-horizon forecasters while
    ensuring data consistency and validation throughout the pipeline.

    Invariants:
        - fit() must be called before predict()
        - Forecaster and preprocessing horizons must match during initialization

    Example:
        Basic forecasting workflow:

        >>> from openstef_models.models.forecasting import BaseForecaster
        >>> from openstef_models.transforms import FeaturePipeline
        >>>
        >>> # Note: This is a conceptual example showing the API structure
        >>> # Real usage requires implemented forecaster classes
        >>> # forecaster = MyForecaster(config=ForecasterConfig(...))
        >>> # preprocessing = FeaturePipeline(horizons=forecaster.config.horizons)
        >>> #
        >>> # Create and train model
        >>> # model = ForecastingModel(
        >>> #     forecaster=forecaster,
        >>> #     preprocessing=preprocessing
        >>> # )
        >>> # model.fit(training_data)
        >>> #
        >>> # Generate forecasts
        >>> # forecasts = model.predict(new_data)
    """

    preprocessing: FeaturePipeline
    forecaster: BaseForecaster | BaseHorizonForecaster
    postprocessing: ForecastTransformPipeline
    target_column: str

    def __init__(
        self,
        forecaster: BaseForecaster | BaseHorizonForecaster,
        preprocessing: FeaturePipeline | None = None,
        postprocessing: ForecastTransformPipeline | None = None,
        target_column: str = "load",
    ):
        """Initialize the forecasting model with required and optional components.

        Args:
            forecaster: The underlying forecasting algorithm (single or multi-horizon).
            preprocessing: Feature engineering pipeline. If None, creates default pipeline
                matching forecaster horizons.
            postprocessing: Result transformation pipeline. If None, creates empty pipeline.
            target_column: Name of the target variable column in datasets.

        Raises:
            ConfigurationError: If forecaster and preprocessing configurations are incompatible.
        """
        preprocessing = preprocessing or FeaturePipeline(horizons=forecaster.config.horizons)

        if forecaster.config.horizons != preprocessing.horizons:
            message = (
                f"The forecaster horizons ({forecaster.config.horizons}) do not match the "
                f"preprocessing horizons ({preprocessing.horizons})."
            )
            raise ConfigurationError(message)

        self.preprocessing = preprocessing
        self.forecaster = forecaster
        self.postprocessing = postprocessing or ForecastTransformPipeline()
        self.target_column = target_column

    def _prepare_input_data(
        self,
        dataset: VersionedTimeSeriesDataset | TimeSeriesDataset,
        forecast_start: datetime | None = None,
    ) -> dict[LeadTime, ForecastInputDataset]:
        input_data = self.preprocessing.transform(dataset=dataset)
        return {
            key: ForecastInputDataset.from_timeseries_dataset(
                dataset=value,
                target_column=self.target_column,
                forecast_start=forecast_start,
            )
            for key, value in input_data.items()
        }

    @property
    def is_fitted(self) -> bool:
        """Check if the underlying forecaster has been trained.

        Returns:
            True if the forecaster is fitted and ready for prediction.
        """
        return self.forecaster.is_fitted

    def fit(self, dataset: VersionedTimeSeriesDataset | TimeSeriesDataset):
        """Train the forecasting model on the provided dataset.

        Fits the preprocessing pipeline and underlying forecaster. Handles both
        single-horizon and multi-horizon forecasters appropriately.

        Args:
            dataset: Historical time series data with features and target values.

        Raises:
            UnreachableStateError: If no data is available for horizon forecasting,
                indicating a violated invariant in the preprocessing pipeline.
        """
        # Fit the feature engineering transforms
        self.preprocessing.fit(dataset=dataset)

        # Transform the input data to a valid forecast input
        input_data = self._prepare_input_data(dataset=dataset)

        # Fit the model
        if isinstance(self.forecaster, BaseForecaster):
            self.forecaster.fit(input_data=input_data)
        else:
            horizon_input_data = next(iter(input_data.values()), None)
            if horizon_input_data is None:
                raise UnreachableStateError("No data available for horizon forecasting.")

            self.forecaster.fit_horizon(input_data=horizon_input_data)

    def predict(
        self, dataset: VersionedTimeSeriesDataset | TimeSeriesDataset, forecast_start: datetime | None = None
    ) -> ForecastDataset:
        """Generate forecasts using the trained model.

        Transforms input data through the preprocessing pipeline, generates predictions
        using the underlying forecaster, and applies postprocessing transformations.

        Args:
            dataset: Input time series data for generating forecasts.
            forecast_start: Starting time for forecasts. If None, uses dataset end time.

        Returns:
            Processed forecast dataset with predictions and uncertainty estimates.

        Raises:
            NotFittedError: If the model hasn't been trained yet.
            UnreachableStateError: If no data is available for horizon forecasting,
                indicating a violated invariant in the preprocessing pipeline.
        """
        if not self.is_fitted:
            raise NotFittedError(type(self.forecaster).__name__)

        # Transform the input data to a valid forecast input
        input_data = self._prepare_input_data(dataset=dataset, forecast_start=forecast_start)

        # Generate predictions
        if isinstance(self.forecaster, BaseForecaster):
            raw_forecasts = self.forecaster.predict(input_data=input_data)
        else:
            horizon_input_data = next(iter(input_data.values()), None)
            if horizon_input_data is None:
                raise UnreachableStateError("No data available for horizon forecasting.")

            raw_forecasts = self.forecaster.predict_horizon(input_data=horizon_input_data)

        return self.postprocessing.transform(raw_forecasts)

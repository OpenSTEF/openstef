from datetime import timedelta
from functools import partial
from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd
from pydantic import Field
from openstef_beam.benchmarking.benchmarks.liander2024 import Liander2024TargetProvider
from openstef_core.base_model import BaseConfig
from openstef_core.datasets import ForecastDataset
from openstef_core.datasets.versioned_timeseries_dataset import (
    VersionedTimeSeriesDataset,
)
from openstef_core.types import LeadTime, Quantile
from openstef_models.models.forecasting_model import ForecastingModel
from openstef_models.presets import (
    ForecastingWorkflowConfig,
    create_forecasting_workflow,
)
from openstef_models.workflows.custom_forecasting_workflow import (
    CustomForecastingWorkflow,
)
from tqdm import tqdm


class GreedyBacktestConfig(BaseConfig):
    """Simplified backtesting for efficient parameter tuning."""

    forecasting_model: ForecastingModel = Field(description="The forecasting model to be evaluated in backtesting.")
    horizon: LeadTime = Field(description="Forecasting horizon for backtesting.")
    training_data_length: timedelta = Field(
        description="Length of historical data used for training at each retraining point.",
        default=timedelta(days=90),
    )
    model_train_interval: timedelta = Field(
        default=timedelta(days=31),
        description="Interval at which the model is retrained during backtesting.",
    )
    max_lagged_features: timedelta = Field(
        default=timedelta(days=14),
        description="Maximum lagged features to consider from historical data.",
    )


class GreedyBackTestPipeline:
    """Greedy backtest pipeline for parameter tuning."""

    def __init__(self, config: GreedyBacktestConfig):
        """Initialize the greedy backtest pipeline."""
        self.config = config

    def _split_data(
        self, dataset: VersionedTimeSeriesDataset
    ) -> tuple[list[VersionedTimeSeriesDataset], list[VersionedTimeSeriesDataset]]:
        training_length = self.config.training_data_length
        model_train_interval = self.config.model_train_interval
        horizon = self.config.horizon.value
        max_lags = self.config.max_lagged_features

        n_splits = (
            int((dataset.index.max() - dataset.index.min() - training_length - horizon) / model_train_interval) + 1
        )
        training_stamps = [pd.Timestamp(dataset.index.min() + i * model_train_interval) for i in range(n_splits)]

        training_sets: list[VersionedTimeSeriesDataset] = []
        prediction_sets: list[VersionedTimeSeriesDataset] = []

        for train_start in training_stamps:
            train_end = train_start + training_length
            test_start = train_end - max_lags
            test_end = min(train_end + model_train_interval, dataset.index.max())  # type : ignore

            training_set = dataset.filter_by_range(train_start, train_end)
            testing_set = dataset.filter_by_range(test_start, test_end)

            training_sets.append(training_set)
            prediction_sets.append(testing_set)

        return training_sets, prediction_sets

    def run(
        self, predictors: VersionedTimeSeriesDataset, ground_truth: VersionedTimeSeriesDataset, start=None, end=None
    ) -> ForecastDataset:
        """Run the greedy backtest pipeline.

        Args:
            predictors: The predictor dataset.
            ground_truth: The ground truth dataset.
            start: (Included for compatibility, not used)
            end:  (Included for compatibility, not used)

        Returns:
            ForecastDataset: The forecast dataset containing predictions and ground truth.
        """
        dataset = VersionedTimeSeriesDataset.concat([ground_truth, predictors], mode="outer").select_version()

        training_datasets, prediction_datasets = self._split_data(dataset)
        pbar = tqdm(total=sum(len(d.index) for d in prediction_datasets), smoothing=0.0)
        predictions = []
        for training_data, prediction_data in zip(training_datasets, prediction_datasets, strict=False):
            self.config.forecasting_model.preprocessing.fit(training_data)
            training_input = self.config.forecasting_model.prepare_input(
                data=training_data,
            )
            target_dropna = partial(pd.DataFrame.dropna, subset=[training_input.target_column])  # pyright: ignore[reportUnknownMemberType]
            training_input = training_input.pipe_pandas(target_dropna)

            prediction_input = self.config.forecasting_model.prepare_input(
                data=prediction_data,
            )

            forecaster = self.config.forecasting_model.forecaster
            forecaster.fit(training_input)
            predictions.append(forecaster.predict(prediction_input))
            pbar.update(len(prediction_data.index))

        target = ground_truth.select_version()

        return ForecastDataset(
            data=pd.concat([pred.data for pred in predictions], axis=0).join(target.data, how="inner").dropna(),
            sample_interval=predictions[0].sample_interval,
            target_column=target.data.columns[0],
        )

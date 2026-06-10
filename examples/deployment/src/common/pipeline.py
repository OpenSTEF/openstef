# SPDX-FileCopyrightText: 2026 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""The OpenSTEF forecasting pipeline shared by all three deployment examples.

This is the *real* code path: it assembles the data fetched from the mocked
:mod:`common.services` into the :class:`~openstef_core.datasets.TimeSeriesDataset` OpenSTEF
expects, slices it for training vs prediction, and builds the workflow. The ``fit`` /
``predict`` calls themselves stay in each orchestrator so you can see exactly where the
library is invoked.

``input_data`` fetches once (enough history to train, plus weather over the forecast
horizon); ``training_view`` and ``prediction_view`` slice it. The Dagster example uses these
directly as assets; the Airflow and Celery examples use the ``*_dataset`` convenience wrappers.
"""

from __future__ import annotations

import logging

from common import services
from common.config import Settings
from openstef_core.datasets import TimeSeriesDataset
from openstef_models.presets import create_forecasting_workflow
from openstef_models.workflows.custom_forecasting_workflow import CustomForecastingWorkflow

logger = logging.getLogger("openstef.deployment")


def _combine(load: TimeSeriesDataset, weather: TimeSeriesDataset) -> TimeSeriesDataset:
    """Join the measurement and weather series into a single dataset (left join on weather)."""
    combined = weather.copy_with(weather.data.join(load.data, how="left"), is_sorted=True)
    logger.info("assembled dataset: %d rows, %d features", len(combined.data), len(combined.feature_names))
    return combined


def input_data(target: str, *, settings: Settings) -> TimeSeriesDataset:
    """Fetch + combine everything needed for both training and forecasting (a single fetch).

    Load is fetched up to the reference time (no future actuals); weather extends over the
    forecast horizon, exactly like a real weather forecast.
    """
    now = settings.reference_time
    start = now - settings.train_window
    load = services.fetch_load_measurements(target, start=start, end=now, settings=settings)
    weather = services.fetch_weather_forecast(target, start=start, end=now + settings.horizon.value, settings=settings)
    return _combine(load, weather)


def training_view(data: TimeSeriesDataset, *, settings: Settings) -> TimeSeriesDataset:
    """Slice the training window: history ending at the reference time."""
    now = settings.reference_time
    return data.filter_by_range(start=now - settings.train_window, end=now)


def prediction_view(data: TimeSeriesDataset, *, settings: Settings) -> TimeSeriesDataset:
    """Slice the prediction input: recent history plus weather over the forecast horizon."""
    now = settings.reference_time
    return data.filter_by_range(start=now - settings.forecasting.predict_history, end=now + settings.horizon.value)


def training_dataset(target: str, *, settings: Settings) -> TimeSeriesDataset:
    """Convenience: fetch and slice the training dataset for ``target`` in one call."""
    return training_view(input_data(target, settings=settings), settings=settings)


def prediction_dataset(target: str, *, settings: Settings) -> TimeSeriesDataset:
    """Convenience: fetch and slice the prediction dataset for ``target`` in one call."""
    return prediction_view(input_data(target, settings=settings), settings=settings)


def build_workflow(target: str, *, settings: Settings) -> CustomForecastingWorkflow:
    """Create the OpenSTEF workflow for ``target`` (wired to the shared MLflow store)."""
    return create_forecasting_workflow(settings.config_for(target))

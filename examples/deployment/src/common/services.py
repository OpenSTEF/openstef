# SPDX-FileCopyrightText: 2026 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Mocked external systems for the deployment examples.

A real deployment fetches measurements from one system, weather forecasts from another, and
publishes results to a third. These functions stand in for those systems using the Liander
benchmark dataset so the examples run with no infrastructure — **replace each with your own
integration**. They speak OpenSTEF's :class:`~openstef_core.datasets.TimeSeriesDataset` /
:class:`~openstef_core.datasets.ForecastDataset` types, exactly like the real ones would.
"""

from __future__ import annotations

import logging
from datetime import datetime
from functools import lru_cache
from pathlib import Path

from common.config import Settings, model_id_for
from openstef_core.datasets import ForecastDataset, TimeSeriesDataset
from openstef_core.testing import load_liander_dataset

logger = logging.getLogger("openstef.deployment")


@lru_cache(maxsize=16)
def _benchmark_data(target: str, dataset_dir: Path) -> TimeSeriesDataset:
    """Download + cache the Liander data for a target (stands in for your data lake)."""
    return load_liander_dataset(target=target, local_dir=dataset_dir)


def fetch_load_measurements(target: str, *, start: datetime, end: datetime, settings: Settings) -> TimeSeriesDataset:
    """Stand-in for your metering / SCADA system: the measured load between ``start`` and ``end``.

    In production this queries your measurement database. It returns data only up to ``end`` —
    the future load is exactly what we are forecasting.
    """
    measurements = _benchmark_data(target, settings.dataset_dir).filter_by_range(start=start, end=end)
    measurements = measurements.select_features([settings.forecasting.target_column])
    logger.info("fetched %d load measurements for %s (%s .. %s)", len(measurements.data), target, start, end)
    return measurements


def fetch_weather_forecast(target: str, *, start: datetime, end: datetime, settings: Settings) -> TimeSeriesDataset:
    """Stand-in for a weather provider (Open-Meteo, KNMI, ...): exogenous features over the window.

    In production this calls a weather API. The data extends past "now" into the forecast
    horizon, just like a real weather forecast.
    """
    window = _benchmark_data(target, settings.dataset_dir).filter_by_range(start=start, end=end)
    weather_features = [name for name in window.feature_names if name != settings.forecasting.target_column]
    weather = window.select_features(weather_features)
    logger.info(
        "fetched weather forecast for %s: %d rows, %d features", target, len(weather.data), len(weather_features)
    )
    return weather


def publish_forecast(forecast: ForecastDataset, target: str, *, settings: Settings) -> Path:
    """Stand-in for delivering the forecast downstream (database, API, message bus).

    Here it writes a parquet file to the output directory and returns its path.
    """
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    path = settings.output_dir / f"{model_id_for(target)}.parquet"
    forecast.data.to_parquet(path)
    logger.info("published forecast for %s -> %s (%d rows)", target, path, len(forecast.data))
    return path

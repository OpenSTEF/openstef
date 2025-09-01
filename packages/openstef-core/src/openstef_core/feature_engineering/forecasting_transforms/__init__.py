# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Forecasting feature transforms for time series data.

This module provides transforms specifically designed for forecasting workflows,
including feature scaling, trend analysis, and feature clipping utilities that
prepare time series data for machine learning models.
"""
from .rolling_aggregate_transform import RollingAggregateTransform


__all__ = ["RollingAggregateTransform"]
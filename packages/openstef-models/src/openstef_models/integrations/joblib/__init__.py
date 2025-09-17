"""Joblib-based model storage integration.

Provides local file-based model persistence using joblib for serialization.
This integration allows storing and loading ForecastingModel instances on
the local filesystem, making it suitable for development, testing, and
single-machine deployments.
"""

# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

from .storage import LocalModelStorage

__all__ = ["LocalModelStorage"]

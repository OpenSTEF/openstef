# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Model implementations and model interfaces for OpenSTEF.

Contains core forecasting model interfaces and package-level convenience
imports.
"""

from .component_splitting_model import ComponentSplittingModel
from .forecasting_model import ForecastingModel

__all__ = [
    "ComponentSplittingModel",
    "ForecastingModel",
]

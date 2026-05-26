# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Pipeline orchestrations for OpenSTEF.

High-level training, forecasting workflows.
"""

from .custom_component_split_workflow import CustomComponentSplitWorkflow
from .custom_forecasting_workflow import CustomForecastingWorkflow

__all__ = [
    "CustomComponentSplitWorkflow",
    "CustomForecastingWorkflow",
]

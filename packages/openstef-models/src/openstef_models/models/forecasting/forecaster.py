# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Re-exports of core forecasting interfaces from openstef-core.

The canonical definitions of Forecaster and ForecasterConfig
now live in ``openstef_core.mixins.forecaster``. This module re-exports them for
backwards compatibility.
"""

# TODO: Remove... Backwards compat not needed

from openstef_core.mixins.forecaster import (
    Forecaster,
    ForecasterConfig,
)

__all__ = [
    "Forecaster",
    "ForecasterConfig",
]

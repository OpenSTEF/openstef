# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Forecasting implementations for OpenSTEF models.

Concrete forecaster implementations for different ML frameworks.
The base `Forecaster` and `ForecasterConfig` interfaces live in
``openstef_core.mixins.forecaster``.

Implementations:
    - constant_median_forecaster: Simple baseline forecaster using historical medians
    - multi_horizon_adapter: Adapter pattern for converting single to multi-horizon forecasters
"""

# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Explainability utilities for OpenSTEF.

Tools for feature importance, attribution and model interpretation. Keep
integration adapters small; heavy dependencies (e.g. SHAP) should be
optional extras and imported lazily.
"""

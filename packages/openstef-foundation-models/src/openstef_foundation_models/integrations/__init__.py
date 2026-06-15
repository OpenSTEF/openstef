# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Integrations with optional external frameworks.

This package is import-light: each integration module carries its own optional
dependency (e.g. :mod:`openstef_foundation_models.integrations.backtesting`
requires the ``benchmarking`` extra) and must be imported directly so that
importing this package never pulls in a dependency you have not installed.
"""

# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Core types and utilities for OpenSTEF.

Lightweight package containing foundational abstractions used across the
project (dataset types, model state containers, common exceptions and
small helpers). Keep API minimal and stable so higher-level packages can
rely on well-typed primitives.

Design highlights
- Typed primitives: prefer small, well-documented classes (e.g.
  TimeSeriesDataset, ModelState) that encapsulate validation and I/O.
- Minimal dependencies: core should remain dependency-light to ease
  reuse across tooling.
"""

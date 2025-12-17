# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Component splitting models for energy data analysis.

This package provides models and utilities for splitting energy time series
into different components (solar, wind, etc.). Component splitters
help analyze energy sources by decomposing total consumption into constituent
parts based on various algorithms and known ratios.
"""

from openstef_models.models import component_splitting

from .component_splitter import ComponentSplitter, ComponentSplitterConfig

__all__ = ["ComponentSplitter", "ComponentSplitterConfig", "component_splitting"]

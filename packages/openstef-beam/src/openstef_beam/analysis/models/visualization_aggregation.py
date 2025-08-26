# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Analysis aggregation models for visualization grouping.

This module defines aggregation levels and scopes that control how evaluation
reports are grouped and compared in visualizations.
"""

from enum import StrEnum

from openstef_beam.analysis.models.target_metadata import GroupName, RunName, TargetName
from openstef_core.base_model import BaseConfig


class AnalysisAggregation(StrEnum):
    """Defines the aggregation levels for visualizations.

    Each aggregation level determines how evaluation reports are grouped and
    compared in visualizations, enabling different analytical perspectives.

    Members:
    - NONE ("none"): Single run, single target - individual performance analysis
    - TARGET ("target"): Single run, per target - cross-target comparison (e.g., RMAE per target)
    - GROUP ("group"): Single run, multiple targets - cross-group comparison (e.g., RMAE per group)
    - RUN ("run"): Multiple runs, per target - model comparison on same target for all targets
    - RUN_AND_GROUP ("run_and_group"): Multiple runs, multiple targets - comprehensive comparison matrix
    """

    NONE = "none"
    TARGET = "target"
    GROUP = "group"
    RUN_AND_NONE = "run_and_none"
    RUN_AND_TARGET = "run_and_target"
    RUN_AND_GROUP = "run_and_group"


class AnalysisScope(BaseConfig):
    """Defines the scope context for analysis operations.

    Specifies which targets, groups, and runs are included in an analysis,
    along with the aggregation level for grouping data.
    """

    target_name: TargetName | None = None
    group_name: GroupName | None = None
    run_name: RunName | None = None
    aggregation: AnalysisAggregation

    def __hash__(self) -> int:
        """Enable using AnalysisScope as a dictionary key.

        Returns:
            Hash value based on all scope fields.
        """
        return hash((self.target_name, self.group_name, self.run_name, self.aggregation))


__all__ = ["AnalysisAggregation", "AnalysisScope"]

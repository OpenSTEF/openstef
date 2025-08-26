# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Target metadata models for analysis pipeline.

This module defines the metadata structure that carries target information
through the analysis pipeline, including grouping and filtering context.
"""

from typing import Self

from pydantic import Field

from openstef_beam.evaluation.models import Filtering
from openstef_core.base_model import BaseModel

type TargetName = str
type GroupName = str
type RunName = str


class TargetMetadata(BaseModel):
    """Metadata for a forecasting target in the analysis pipeline.

    Contains essential information about a target including its grouping context
    and lead time filtering criteria. Lead time filtering determines which
    predictions are included based on how far ahead they were made (e.g.,
    1-hour ahead vs 24-hour ahead forecasts).
    """

    name: TargetName = Field(description="Name of the target")
    group_name: GroupName = Field(description="Name of the group this target belongs to, used for grouping in reports")
    filtering: Filtering | None = Field(
        description="Lead time filtering criteria - either AvailableAt (data availability time) "
        "or LeadTime (forecast horizon)"
    )
    limit: float = Field(description="Capacity limit of the target in appropriate units")
    run_name: RunName = Field(description="Name of the run associated with this target")

    def with_filtering(self, filtering: Filtering) -> Self:
        """Returns a copy of the target metadata with different lead time filtering applied.

        Args:
            filtering: New lead time filtering criteria to apply

        Returns:
            New TargetMetadata instance with updated lead time filtering
        """
        return type(self)(
            name=self.name,
            group_name=self.group_name,
            filtering=filtering,
            limit=self.limit,
            run_name=self.run_name,
        )


__all__ = ["GroupName", "RunName", "TargetMetadata", "TargetName"]

# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

from typing import Self

from pydantic import Field

from openstef_beam.evaluation.models import Filtering
from openstef_core.base_model import BaseModel

type TargetName = str
type GroupName = str
type RunName = str


class TargetMetadata(BaseModel):
    """Metadata for a target in the analytics pipeline."""

    name: TargetName = Field(description="Name of the target")
    group_name: GroupName = Field(description="Name of the group this target belongs to, used for grouping in reports")
    filtering: Filtering | None = Field(
        description="Filtering criteria for the target, can be either AvailableAt or LeadTime"
    )
    limit: float = Field(description="Capacity limit of the target in appropriate units")
    run_name: RunName = Field(description="Name of the run associated with this target")

    def with_filtering(self, filtering: Filtering) -> Self:
        """Returns a copy of the target metadata with the specified filtering applied."""
        return type(self)(
            name=self.name,
            group_name=self.group_name,
            filtering=filtering,
            limit=self.limit,
            run_name=self.run_name,
        )


__all__ = ["GroupName", "RunName", "TargetMetadata", "TargetName"]

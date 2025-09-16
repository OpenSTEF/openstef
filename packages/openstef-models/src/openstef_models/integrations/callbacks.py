# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

from typing import TYPE_CHECKING

from openstef_core.datasets import TimeSeriesDataset, VersionedTimeSeriesDataset
from openstef_core.datasets.validated_datasets import ForecastDataset

if TYPE_CHECKING:
    from openstef_models.workflows.forecasting_workflow import ForecastingWorkflow


class ForecastingCallback:
    def on_before_fit(self, pipeline: "ForecastingWorkflow", dataset: VersionedTimeSeriesDataset | TimeSeriesDataset):
        pass

    def on_after_fit(self, pipeline: "ForecastingWorkflow", dataset: VersionedTimeSeriesDataset | TimeSeriesDataset):
        pass

    def on_before_predict(
        self, pipeline: "ForecastingWorkflow", dataset: VersionedTimeSeriesDataset | TimeSeriesDataset
    ):
        pass

    def on_after_predict(
        self,
        pipeline: "ForecastingWorkflow",
        dataset: VersionedTimeSeriesDataset | TimeSeriesDataset,
        forecasts: ForecastDataset,
    ):
        pass

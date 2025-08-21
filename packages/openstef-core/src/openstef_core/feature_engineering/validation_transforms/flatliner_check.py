# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

from openstef_core.datasets import TimeSeriesDataset
from openstef_core.transforms.base import TimeSeriesTransform


class FlatlinerCheck(TimeSeriesTransform):
    def fit(self, data: TimeSeriesDataset) -> None:
        pass

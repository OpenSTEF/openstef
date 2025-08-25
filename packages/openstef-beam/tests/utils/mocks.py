# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

from typing import cast

from openstef_beam.evaluation.metric_providers import MetricProvider, QuantileMetricsDict
from openstef_beam.evaluation.models import EvaluationSubset


class DummyMetricProvider(MetricProvider):
    """Returns a constant metric for every call."""

    value: float = 1.0

    def __call__(self, subset: EvaluationSubset) -> QuantileMetricsDict:
        # Return metrics for global since test is not using actual quantile data
        return cast(QuantileMetricsDict, {"global": {"dummy_metric": self.value}})


class MockFigure:
    def __init__(self):
        self.data = []
        self.layout = type("Layout", (), {"title": type("Title", (), {"text": "Test Plot"})})()

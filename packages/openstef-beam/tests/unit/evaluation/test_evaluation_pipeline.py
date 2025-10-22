# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

from datetime import timedelta

import pandas as pd
import pytest

from openstef_beam.evaluation.evaluation_pipeline import EvaluationConfig, EvaluationPipeline
from openstef_beam.evaluation.models import EvaluationReport, EvaluationSubsetReport, SubsetMetric, Window
from openstef_core.datasets import TimeSeriesDataset, VersionedTimeSeriesDataset
from openstef_core.exceptions import MissingColumnsError
from openstef_core.types import AvailableAt, LeadTime, Quantile
from tests.utils.mocks import DummyMetricProvider


@pytest.fixture
def predictions_dataset() -> TimeSeriesDataset:
    return TimeSeriesDataset(
        data=pd.DataFrame(
            {
                "quantile_P50": [1, 2, 3, 4],
                "quantile_P90": [1, 2, 3, 4],
                "available_at": pd.date_range("2020-01-01T00:00", periods=4, freq="h") - timedelta(hours=48),
            },
            index=pd.date_range("2020-01-01T00:00", periods=4, freq="h"),
        ),
        sample_interval=timedelta(hours=1),
    )


@pytest.fixture
def minimal_ground_truth_dataset() -> VersionedTimeSeriesDataset:
    return VersionedTimeSeriesDataset.from_dataframe(
        data=pd.DataFrame(
            {
                "target": [1, 2, 3, 4],
                "available_at": pd.date_range("2020-01-01T00:00", periods=4, freq="h"),
            },
            index=pd.date_range("2020-01-01T00:00", periods=4, freq="h"),
        ),
        sample_interval=timedelta(hours=1),
    )


@pytest.fixture
def minimal_config() -> EvaluationConfig:
    return EvaluationConfig(
        available_ats=[AvailableAt.from_string("D-1T00:00")],
        lead_times=[LeadTime.from_string("PT1H")],
        windows=[Window(lag=timedelta(hours=0), size=timedelta(hours=2), stride=timedelta(hours=1))],
    )


def test_run_raises_on_missing_quantile_column(
    predictions_dataset: VersionedTimeSeriesDataset, minimal_ground_truth_dataset: VersionedTimeSeriesDataset
):
    # Arrange
    config = EvaluationConfig(
        available_ats=[AvailableAt.from_string("D-1T00:00")],
        lead_times=[LeadTime.from_string("PT1H")],
        windows=[Window(lag=timedelta(hours=0), size=timedelta(hours=2), stride=timedelta(hours=1))],
    )
    pipeline = EvaluationPipeline(
        config=config,
        window_metric_providers=[DummyMetricProvider()],
        global_metric_providers=[DummyMetricProvider()],
        quantiles=[Quantile(0.5), Quantile(0.9), Quantile(0.1)],  # 0.1 is missing in predictions
    )

    # Act / Assert
    with pytest.raises(MissingColumnsError, match="quantile_P10"):
        pipeline.run(
            predictions=predictions_dataset,
            ground_truth=minimal_ground_truth_dataset,
            target_column="target",
        )


def test_run_returns_evaluation_report(
    minimal_config: EvaluationConfig,
    predictions_dataset: VersionedTimeSeriesDataset,
    minimal_ground_truth_dataset: VersionedTimeSeriesDataset,
):
    # Arrange
    pipeline = EvaluationPipeline(
        config=minimal_config,
        window_metric_providers=[DummyMetricProvider(value=2.0)],
        global_metric_providers=[DummyMetricProvider(value=3.0)],
        quantiles=[Quantile(0.5), Quantile(0.9)],
    )

    # Act
    report = pipeline.run(
        predictions=predictions_dataset, ground_truth=minimal_ground_truth_dataset, target_column="target"
    )

    # Assert
    assert isinstance(report, EvaluationReport)
    assert len(report.subset_reports) == 2

    for subset_report in report.subset_reports:
        assert isinstance(subset_report, EvaluationSubsetReport)
        assert all(isinstance(m, SubsetMetric) for m in subset_report.metrics)
        # 4 hourly datapoints mean 2 computed windows and 1 global window
        assert sum(1 for m in subset_report.metrics if m.window == "global") == 1
        assert sum(1 for m in subset_report.metrics if m.window != "global") == 2
        # Check that dummy metrics are present
        assert any(m.metrics.get("global", {}).get("dummy_metric") == 2.0 for m in subset_report.metrics)
        assert all(
            m.metrics.get("global", {}).get("dummy_metric") == 3.0
            for m in subset_report.metrics
            if m.window == "global"
        )

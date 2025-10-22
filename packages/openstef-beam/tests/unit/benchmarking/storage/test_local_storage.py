# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

import tempfile
from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest

from openstef_beam.analysis import AnalysisOutput, AnalysisScope, VisualizationOutput
from openstef_beam.analysis.models import AnalysisAggregation
from openstef_beam.benchmarking.models import BenchmarkTarget
from openstef_beam.benchmarking.storage import LocalBenchmarkStorage
from openstef_beam.evaluation import EvaluationReport, EvaluationSubsetReport, Filtering, SubsetMetric
from openstef_core.datasets import ForecastDataset, TimeSeriesDataset
from openstef_core.types import AvailableAt, LeadTime


def assert_directory_and_file_exist(expected_path: Path) -> None:
    """Helper function to check if a file and its parent directory exist."""
    assert expected_path.parent.exists(), f"Parent directory {expected_path.parent} does not exist"
    assert expected_path.exists(), f"File {expected_path} does not exist"


def assert_can_roundtrip_data[T](
    storage: LocalBenchmarkStorage,
    target: BenchmarkTarget,
    save_func: Callable[[BenchmarkTarget, T], None],
    load_func: Callable[[BenchmarkTarget], T],
    data: T,
) -> T:
    """Helper function to test save/load roundtrip functionality."""
    save_func(target, data)
    return load_func(target)


@pytest.fixture
def target() -> BenchmarkTarget:
    """Create a test target."""
    return BenchmarkTarget(
        name="test_target",
        description="Test target",
        latitude=0.0,
        longitude=0.0,
        limit=100.0,
        train_start=pd.Timestamp("2022-12-25"),
        benchmark_start=pd.Timestamp("2023-01-01"),
        benchmark_end=pd.Timestamp("2023-01-10"),
    )


@pytest.fixture
def predictions() -> TimeSeriesDataset:
    """Create test predictions."""
    return TimeSeriesDataset(
        data=pd.DataFrame(
            {
                "value": [1.0, 2.0],
                "available_at": pd.date_range("2023-01-01", periods=2, freq="1h"),
            },
            index=pd.DatetimeIndex(pd.date_range("2023-01-07", periods=2, freq="1h")),
        ),
        sample_interval=timedelta(hours=1),
    )


@pytest.fixture
def evaluation_report() -> EvaluationReport:
    """Create a test evaluation report."""
    return EvaluationReport(
        subset_reports=[
            EvaluationSubsetReport(
                filtering=AvailableAt.from_string("D-1T06:00"),
                subset=ForecastDataset(
                    data=pd.DataFrame(
                        data={"quantile_P50": [1.0, 2.0], "load": [3.0, 4.0]},
                        index=pd.date_range("2023-01-07", periods=2, freq="1h"),
                    ),
                    sample_interval=timedelta(hours=1),
                ),
                metrics=[
                    SubsetMetric(
                        timestamp=datetime.fromisoformat("2023-01-07T00:00:00+00:00"),
                        window="global",
                        metrics={"global": {"rmae": 0.1}},
                    )
                ],
            )
        ]
    )


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def local_storage(temp_dir: Path) -> LocalBenchmarkStorage:
    """Create local storage instance."""
    return LocalBenchmarkStorage(base_path=temp_dir)


@pytest.mark.parametrize(
    ("custom_filename", "expected_filename"),
    [
        pytest.param("predictions.parquet", "predictions.parquet", id="default"),
        pytest.param("custom_predictions.parquet", "custom_predictions.parquet", id="custom"),
    ],
)
def test_predictions_path_construction(
    temp_dir: Path, target: BenchmarkTarget, custom_filename: str, expected_filename: str
):
    """Test predictions path construction with different filenames."""
    # Arrange
    storage = LocalBenchmarkStorage(base_path=temp_dir, predictions_filename=custom_filename)

    # Act
    predictions_path = storage.get_predictions_path_for_target(target)

    # Assert
    expected_path = temp_dir / "backtest" / target.group_name / "test_target" / expected_filename
    assert predictions_path == expected_path


def test_save_and_load_backtest_output(
    local_storage: LocalBenchmarkStorage, target: BenchmarkTarget, predictions: TimeSeriesDataset
):
    """Test saving and loading backtest output."""
    # Arrange & Act
    local_storage.save_backtest_output(target, predictions)

    # Assert
    assert local_storage.has_backtest_output(target)

    loaded_predictions = local_storage.load_backtest_output(target)
    pd.testing.assert_frame_equal(loaded_predictions.data, predictions.data, check_freq=False)
    assert loaded_predictions.sample_interval == predictions.sample_interval


def test_save_and_load_evaluation_output(
    local_storage: LocalBenchmarkStorage, target: BenchmarkTarget, evaluation_report: EvaluationReport
):
    """Test saving and loading evaluation output."""
    # Arrange & Act
    local_storage.save_evaluation_output(target, evaluation_report)

    # Assert
    assert local_storage.has_evaluation_output(target)

    loaded_evaluation = local_storage.load_evaluation_output(target)
    assert len(loaded_evaluation.subset_reports) == len(evaluation_report.subset_reports)
    assert loaded_evaluation.subset_reports[0].metrics == evaluation_report.subset_reports[0].metrics


def test_save_backtest_creates_directory_structure(
    local_storage: LocalBenchmarkStorage, target: BenchmarkTarget, predictions: TimeSeriesDataset
):
    """Test that saving backtest output creates proper directory structure."""
    # Arrange
    expected_path = local_storage.get_predictions_path_for_target(target)

    # Act
    local_storage.save_backtest_output(target, predictions)

    # Assert
    assert_directory_and_file_exist(expected_path)


def test_save_evaluation_creates_directory_structure(
    local_storage: LocalBenchmarkStorage, target: BenchmarkTarget, evaluation_report: EvaluationReport
):
    """Test that saving evaluation output creates proper directory structure."""
    # Arrange
    expected_path = local_storage.get_evaluations_path_for_target(target)

    # Act
    local_storage.save_evaluation_output(target, evaluation_report)

    # Assert
    assert expected_path.exists()
    # Verify that the evaluation report was saved correctly by testing roundtrip
    loaded_evaluation = assert_can_roundtrip_data(
        local_storage,
        target,
        local_storage.save_evaluation_output,
        local_storage.load_evaluation_output,
        evaluation_report,
    )
    assert len(loaded_evaluation.subset_reports) == len(evaluation_report.subset_reports)


def test_save_analysis_output_creates_files(local_storage: LocalBenchmarkStorage, target: BenchmarkTarget):
    """Test that analysis output saves visualization files."""
    # Arrange
    mock_output: VisualizationOutput = Mock(spec=VisualizationOutput)
    mock_output.name = "test_chart"
    mock_output.write_html = Mock()
    filter_value = LeadTime(timedelta(days=1))
    visualizations: dict[Filtering, list[VisualizationOutput]] = {filter_value: [mock_output]}
    output = AnalysisOutput(
        scope=AnalysisScope(
            aggregation=AnalysisAggregation.NONE,
            group_name=target.group_name,
            target_name=target.name,
            run_name=None,
        ),
        visualizations=visualizations,
    )

    # Act
    local_storage.save_analysis_output(output)

    # Assert
    expected_dir = local_storage.get_analysis_path(output.scope)
    expected_file = expected_dir / str(filter_value) / "test_chart.html"

    assert expected_dir.exists()
    mock_output.write_html.assert_called_once_with(expected_file)
    assert local_storage.has_analysis_output(output.scope)


def test_has_backtest_output_with_missing_file(local_storage: LocalBenchmarkStorage, target: BenchmarkTarget):
    """Test has_backtest_output returns False when file doesn't exist."""
    # Arrange & Act & Assert
    assert not local_storage.has_backtest_output(target)


def test_has_evaluation_output_with_missing_file(local_storage: LocalBenchmarkStorage, target: BenchmarkTarget):
    """Test has_evaluation_output returns False when file doesn't exist."""
    # Arrange & Act & Assert
    assert not local_storage.has_evaluation_output(target)


def test_has_analysis_output_with_missing_directory(local_storage: LocalBenchmarkStorage, target: BenchmarkTarget):
    """Test has_analysis_output returns False when directory doesn't exist."""
    # Arrange
    scope = AnalysisScope(
        aggregation=AnalysisAggregation.NONE,
        group_name=target.group_name,
        target_name=target.name,
        run_name=None,
    )

    # Act & Assert
    assert not local_storage.has_analysis_output(scope)


def test_path_methods_return_valid_paths(local_storage: LocalBenchmarkStorage, target: BenchmarkTarget):
    """Test that all path methods return valid Path objects and are consistent."""
    # Test predictions path
    predictions_path = local_storage.get_predictions_path_for_target(target)
    assert isinstance(predictions_path, Path)
    assert predictions_path.name == local_storage.predictions_filename
    assert str(target.group_name) in str(predictions_path)
    assert str(target.name) in str(predictions_path)

    # Test evaluations path
    evaluations_path = local_storage.get_evaluations_path_for_target(target)
    assert isinstance(evaluations_path, Path)
    assert str(target.group_name) in str(evaluations_path)
    assert str(target.name) in str(evaluations_path)

    # Test analysis path
    scope = AnalysisScope(
        aggregation=AnalysisAggregation.NONE,
        group_name=target.group_name,
        target_name=target.name,
        run_name=None,
    )
    analysis_path = local_storage.get_analysis_path(scope)
    assert isinstance(analysis_path, Path)

    # All paths should be under the base path
    assert local_storage.base_path in predictions_path.parents
    assert local_storage.base_path in evaluations_path.parents
    assert local_storage.base_path in analysis_path.parents


def test_directory_structure_consistency(local_storage: LocalBenchmarkStorage, target: BenchmarkTarget):
    """Test that directory structure is consistent across different operations."""
    # Get paths
    predictions_path = local_storage.get_predictions_path_for_target(target)
    evaluations_path = local_storage.get_evaluations_path_for_target(target)

    # Both should have the same group and target names in their paths
    assert str(target.group_name) in str(predictions_path)
    assert str(target.group_name) in str(evaluations_path)
    assert str(target.name) in str(predictions_path)
    assert str(target.name) in str(evaluations_path)

    # The directory structure should be predictable
    assert local_storage.backtest_dirname in str(predictions_path)
    assert local_storage.evaluations_dirname in str(evaluations_path)

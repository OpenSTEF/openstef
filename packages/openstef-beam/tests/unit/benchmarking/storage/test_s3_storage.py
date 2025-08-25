# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

import os
import tempfile
import uuid
from collections.abc import Iterator
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest
import s3fs
from moto.server import ThreadedMotoServer

from openstef_beam.benchmarking import S3BenchmarkStorage
from openstef_beam.benchmarking.models import BenchmarkTarget
from openstef_beam.benchmarking.storage import LocalBenchmarkStorage
from openstef_beam.evaluation import EvaluationReport, EvaluationSubsetReport, SubsetMetric
from openstef_beam.evaluation.models import EvaluationSubset
from openstef_core.datasets import TimeSeriesDataset, VersionedTimeSeriesDataset
from openstef_core.types import AvailableAt


@pytest.fixture(scope="session")
def moto_server():
    """Start ThreadedMotoServer for the test session."""
    # Arrange
    os.environ["AWS_ACCESS_KEY_ID"] = "testing"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "testing"
    os.environ["AWS_DEFAULT_REGION"] = "us-east-1"

    server = ThreadedMotoServer(port=0, verbose=False)
    server.start()
    host, port = server.get_host_and_port()
    try:
        yield f"http://{host}:{port}"
    finally:
        server.stop()


@pytest.fixture
def s3_setup(moto_server: str):
    """Setup S3 filesystem pointing to moto server."""
    # Arrange
    fs = s3fs.S3FileSystem(client_kwargs={"endpoint_url": moto_server}, key="testing", secret="testing")
    bucket_name = f"test-bucket-{uuid.uuid4().hex[:8]}"
    fs.makedirs(bucket_name, exist_ok=True)
    return fs, bucket_name


@pytest.fixture
def target():
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
def predictions():
    """Create test predictions."""
    return VersionedTimeSeriesDataset(
        data=pd.DataFrame({
            "value": [1.0, 2.0],
            "timestamp": pd.date_range("2023-01-07", periods=2, freq="1h"),
            "available_at": pd.date_range("2023-01-01", periods=2, freq="1h"),
        }),
        sample_interval=timedelta(hours=1),
    )


@pytest.fixture
def evaluation_report():
    """Create a test evaluation report."""
    index = pd.date_range("2023-01-07", periods=2, freq="1h")
    return EvaluationReport(
        subset_reports=[
            EvaluationSubsetReport(
                filtering=AvailableAt.from_string("D-1T06:00"),
                subset=EvaluationSubset.create(
                    ground_truth=TimeSeriesDataset(
                        data=pd.DataFrame({"value": [1.0, 2.0]}, index=index),
                        sample_interval=timedelta(hours=1),
                    ),
                    predictions=TimeSeriesDataset(
                        data=pd.DataFrame({"value": [1.0, 2.0]}, index=index),
                        sample_interval=timedelta(hours=1),
                    ),
                    index=index,
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
def temp_dir() -> Iterator[Path]:
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def local_storage(temp_dir: Path) -> LocalBenchmarkStorage:
    """Create local storage instance."""
    return LocalBenchmarkStorage(base_path=temp_dir)


@pytest.mark.parametrize(
    "s3_prefix",
    [
        pytest.param("", id="no_prefix"),
        pytest.param("test/prefix", id="with_prefix"),
        pytest.param("test/prefix/", id="prefix_with_slash"),
    ],
)
def test_s3_path_construction(local_storage: LocalBenchmarkStorage, target: BenchmarkTarget, s3_prefix: str):
    """Test S3 path construction with different prefixes."""
    # Arrange
    s3_storage = S3BenchmarkStorage(local_storage=local_storage, bucket_name="test-bucket", s3_prefix=s3_prefix)
    test_file = local_storage.get_predictions_path_for_target(target)
    test_file.parent.mkdir(parents=True)
    test_file.touch()

    # Act
    s3_path = s3_storage._get_s3_path(test_file)

    # Assert
    expected_prefix = s3_prefix.strip("/") if s3_prefix else ""
    # The path should contain the relative path from base_path
    relative_path = test_file.relative_to(local_storage.base_path)
    if expected_prefix:
        expected_path = f"s3://test-bucket/{expected_prefix}/{relative_path}"
    else:
        expected_path = f"s3://test-bucket/{relative_path}"
    assert s3_path == expected_path


@pytest.mark.parametrize(
    ("operation", "method_name"),
    [
        pytest.param("predictions", "save_backtest_output", id="backtest"),
        pytest.param("evaluations", "save_evaluation_output", id="evaluation"),
    ],
)
def test_s3_upload_on_save(
    s3_setup: tuple[s3fs.S3FileSystem, str],
    local_storage: LocalBenchmarkStorage,
    target: BenchmarkTarget,
    predictions: VersionedTimeSeriesDataset,
    evaluation_report: EvaluationReport,
    moto_server: str,
    operation: str,
    method_name: str,
):
    """Test that files are uploaded to S3 during save operations."""
    # Arrange
    fs: s3fs.S3FileSystem
    bucket_name: str
    fs, bucket_name = s3_setup
    s3_storage = S3BenchmarkStorage(
        local_storage=local_storage,
        bucket_name=bucket_name,
        s3_prefix="test-prefix",
        s3fs_kwargs={"client_kwargs": {"endpoint_url": moto_server}, "key": "testing", "secret": "testing"},
    )

    test_data = predictions if operation == "predictions" else evaluation_report
    method = getattr(s3_storage, method_name)

    # Act
    method(target, test_data)

    # Assert
    if operation == "predictions":
        s3_path = f"{bucket_name}/test-prefix/backtest/default/test_target/predictions.parquet"
        assert fs.exists(s3_path)
        # Verify content
        with fs.open(s3_path, "rb") as f:
            uploaded_data = pd.read_parquet(f)
        local_path = local_storage.get_predictions_path_for_target(target)
        local_data = pd.read_parquet(local_path)
        pd.testing.assert_frame_equal(uploaded_data, local_data)
    else:
        s3_path = f"{bucket_name}/test-prefix/evaluation/default/test_target"
        assert fs.exists(s3_path)


def test_save_evaluation_output_uploads_to_s3(
    s3_setup: tuple[s3fs.S3FileSystem, str],
    local_storage: LocalBenchmarkStorage,
    target: BenchmarkTarget,
    evaluation_report: EvaluationReport,
    moto_server: str,
):
    """Test that evaluation output is uploaded to S3."""
    # Arrange
    fs, bucket_name = s3_setup
    s3_storage = S3BenchmarkStorage(
        local_storage=local_storage,
        bucket_name=bucket_name,
        s3_prefix="test-prefix",
        s3fs_kwargs={"client_kwargs": {"endpoint_url": moto_server}, "key": "testing", "secret": "testing"},
    )

    # Act
    s3_storage.save_evaluation_output(target, evaluation_report)

    # Assert
    # Check that evaluation file was uploaded to S3
    s3_path = f"{bucket_name}/test-prefix/evaluation/default/test_target"
    assert fs.exists(s3_path), f"Evaluation file not found at {s3_path}"

    # Verify content integrity by checking local file was created
    local_path = local_storage.get_evaluations_path_for_target(target)
    assert local_path.exists(), "Local evaluation file was not created"


def test_missing_local_files_handling(
    s3_setup: tuple[s3fs.S3FileSystem, str],
    local_storage: LocalBenchmarkStorage,
    target: BenchmarkTarget,
    moto_server: str,
):
    """Test behavior when local files are missing during upload."""
    # Arrange
    fs, bucket_name = s3_setup
    s3_storage = S3BenchmarkStorage(
        local_storage=local_storage,
        bucket_name=bucket_name,
        s3fs_kwargs={"client_kwargs": {"endpoint_url": moto_server}, "key": "testing", "secret": "testing"},
    )

    # Act - attempt to save when no local files exist by not providing any data
    # This should create the target directory but no actual files
    target_dir = local_storage.get_predictions_path_for_target(target).parent
    target_dir.mkdir(parents=True, exist_ok=True)

    # Try to upload non-existent files
    s3_storage._put_path_to_s3(
        local_path=local_storage.get_predictions_path_for_target(target), artifact_name="predictions"
    )

    # Assert - no files should be uploaded to S3
    objects = list(fs.ls(f"{bucket_name}/", detail=False))
    assert len(objects) == 0


def test_load_operations_delegate_to_local_storage(
    local_storage: LocalBenchmarkStorage,
    target: BenchmarkTarget,
    predictions: VersionedTimeSeriesDataset,
    evaluation_report: EvaluationReport,
):
    """Test that load operations delegate to local storage without accessing S3."""
    # Arrange
    # Create S3 storage without S3 credentials to ensure S3 is not accessed
    s3_storage = S3BenchmarkStorage(local_storage=local_storage, bucket_name="test-bucket")

    # Save some data first using local storage directly to avoid S3 upload
    local_storage.save_backtest_output(target, predictions)
    local_storage.save_evaluation_output(target, evaluation_report)

    # Act & Assert
    loaded_predictions = s3_storage.load_backtest_output(target)
    loaded_evaluation = s3_storage.load_evaluation_output(target)

    assert s3_storage.has_backtest_output(target)
    assert s3_storage.has_evaluation_output(target)

    # Verify data integrity
    pd.testing.assert_frame_equal(loaded_predictions.data, predictions.data)
    assert len(loaded_evaluation.subset_reports) == len(evaluation_report.subset_reports)

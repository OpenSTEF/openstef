# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

import logging
from pathlib import Path, PurePosixPath
from typing import Any

from openstef_beam.analysis import AnalysisOutput, AnalysisScope
from openstef_beam.benchmarking.models import BenchmarkTarget
from openstef_beam.benchmarking.storage.base import BenchmarkStorage
from openstef_beam.benchmarking.storage.local_storage import LocalBenchmarkStorage
from openstef_beam.evaluation import EvaluationReport
from openstef_core.datasets import VersionedTimeSeriesDataset

_logger = logging.getLogger(__name__)


class S3BenchmarkStorage(BenchmarkStorage):
    """S3-backed storage implementation that combines local and cloud storage.

    Provides a hybrid approach where benchmark artifacts are first stored locally
    and then automatically synced to S3. All read operations use the local storage,
    while write operations trigger both local storage and S3 upload.

    Requires the s3fs package for S3 operations. Uses the local storage instance
    for file organization and path management, ensuring consistent structure
    between local and S3 storage.
    """

    def __init__(
        self,
        local_storage: LocalBenchmarkStorage,
        bucket_name: str,
        s3_prefix: str = "",
        s3fs_kwargs: dict[str, Any] | None = None,
    ):
        """Initialize S3 storage with local storage delegation.

        Args:
            local_storage: The LocalBenchmarkStorage instance for local file operations.
            bucket_name: Name of the S3 bucket where files will be stored.
            s3_prefix: Optional prefix for all S3 object keys to organize files.
            s3fs_kwargs: Additional keyword arguments passed to S3FileSystem constructor
                for authentication and configuration.

        Raises:
            ImportError: When s3fs package is not installed.
        """
        super().__init__()
        self.local_storage = local_storage
        self.bucket_name = bucket_name
        self.s3_prefix = s3_prefix.strip("/") if s3_prefix else ""
        self.s3fs_kwargs = s3fs_kwargs or {}

        # Lazy import s3fs to keep it as an optional dependency
        try:
            import s3fs

            self.fs = s3fs.S3FileSystem(**self.s3fs_kwargs)
        except ImportError as e:
            _logger.exception("s3fs not installed. Please install with 'pip install s3fs'")
            raise ImportError("s3fs package is required for S3StorageCallback") from e

    def save_backtest_output(self, target: BenchmarkTarget, output: VersionedTimeSeriesDataset) -> None:
        self.local_storage.save_backtest_output(target, output)

        output_path = self.local_storage.get_predictions_path_for_target(target)
        self._put_path_to_s3(local_path=output_path, artifact_name=self._get_s3_path(output_path))

    def load_backtest_output(self, target: BenchmarkTarget) -> VersionedTimeSeriesDataset:
        return self.local_storage.load_backtest_output(target)

    def has_backtest_output(self, target: BenchmarkTarget) -> bool:
        return self.local_storage.has_backtest_output(target)

    def save_evaluation_output(self, target: BenchmarkTarget, output: EvaluationReport) -> None:
        self.local_storage.save_evaluation_output(target, output)

        output_path = self.local_storage.get_evaluations_path_for_target(target)
        self._put_path_to_s3(local_path=output_path, artifact_name=self._get_s3_path(output_path))

    def load_evaluation_output(self, target: BenchmarkTarget) -> EvaluationReport:
        return self.local_storage.load_evaluation_output(target)

    def has_evaluation_output(self, target: BenchmarkTarget) -> bool:
        return self.local_storage.has_evaluation_output(target)

    def save_analysis_output(self, output: AnalysisOutput) -> None:
        self.local_storage.save_analysis_output(output)

        output_dir = self.local_storage.get_analysis_path(output.scope)
        self._put_path_to_s3(local_path=output_dir, artifact_name=self._get_s3_path(output_dir))

    def has_analysis_output(self, scope: AnalysisScope) -> bool:
        return self.local_storage.has_analysis_output(scope)

    def _put_path_to_s3(self, local_path: Path, artifact_name: str) -> None:
        """Upload a local file or directory to S3.

        Args:
            local_path: The local file or directory path to upload.
            artifact_name: The name of the artifact being uploaded.
        """
        if not local_path.exists():
            _logger.warning(
                "Artifact not found locally, skipping S3 sync.",
                extra={"artifact_name": artifact_name, "local_path": str(local_path)},
            )
            return

        s3_path = self._get_s3_path(local_path)
        if local_path.is_dir():
            _logger.info(
                "Syncing directory to S3",
                extra={"local_path": str(local_path), "s3_path": s3_path, "artifact_name": artifact_name},
            )
            self.fs.put(str(local_path), s3_path, recursive=True)  # type: ignore[reportUnknownMemberType]
        else:
            _logger.info(
                "Syncing file to S3",
                extra={"local_path": str(local_path), "s3_path": s3_path, "artifact_name": artifact_name},
            )
            self.fs.put(str(local_path), s3_path)  # type: ignore[reportUnknownMemberType]

    def _get_s3_path(self, local_path: Path) -> str:
        """Construct the S3 path for a given local path.

        Uses pathlib for cleaner path manipulation and proper S3 URI construction.
        """
        # Get relative path from output directory
        relative_path = local_path.relative_to(self.local_storage.base_path)

        # Use PurePosixPath for S3 path construction (S3 uses forward slashes)
        s3_path_parts: list[str] = []
        if self.s3_prefix:
            s3_path_parts.append(self.s3_prefix)
        s3_path_parts.extend(relative_path.parts)

        # Construct the S3 object key using PurePosixPath
        s3_key = str(PurePosixPath(*s3_path_parts))

        # Return the full S3 URI
        return f"s3://{self.bucket_name}/{s3_key}"


__all__ = ["S3BenchmarkStorage"]

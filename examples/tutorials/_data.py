# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Shared data loading utilities for OpenSTEF tutorials.

Downloads and combines the Liander 2024 benchmark dataset from HuggingFace Hub
into a ready-to-use TimeSeriesDataset.
"""

from pathlib import Path

from huggingface_hub import hf_hub_download  # pyright: ignore[reportUnknownVariableType]

from openstef_core.datasets import TimeSeriesDataset, VersionedTimeSeriesDataset

__all__ = ["load_liander_dataset"]

_DEFAULT_REPO_ID = "OpenSTEF/liander2024-energy-forecasting-benchmark"
_DEFAULT_TARGET = "mv_feeder/OS Gorredijk"


def load_liander_dataset(
    *,
    target: str = _DEFAULT_TARGET,
    repo_id: str = _DEFAULT_REPO_ID,
    local_dir: Path = Path("./liander_dataset"),
    extra_files: list[str] | None = None,
) -> TimeSeriesDataset:
    """Download and combine the Liander benchmark dataset into a single TimeSeriesDataset.

    Downloads load measurements, weather forecasts, electricity prices, and standard load
    profiles from HuggingFace Hub, then combines them via left join.

    Args:
        target: Sub-path within the repo identifying the installation (e.g. ``"mv_feeder/OS Gorredijk"``).
        repo_id: HuggingFace dataset repository ID.
        local_dir: Local directory for caching downloaded files.
        extra_files: Additional parquet files to download and include (paths relative to repo root).

    Returns:
        Combined dataset with all features aligned by timestamp.
    """
    files_to_download = [
        f"load_measurements/{target}.parquet",
        f"weather_forecasts_versioned/{target}.parquet",
        "EPEX.parquet",
        "profiles.parquet",
        *(extra_files or []),
    ]

    for filename in files_to_download:
        hf_hub_download(  # pyright: ignore[reportCallIssue]
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
            local_dir=local_dir,
            local_dir_use_symlinks=False,
        )

    datasets = [VersionedTimeSeriesDataset.read_parquet(local_dir / f) for f in files_to_download]
    return VersionedTimeSeriesDataset.concat(datasets, mode="left").select_version()

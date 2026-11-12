# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

from importlib.metadata import PackageNotFoundError

import pytest
from packaging.version import InvalidVersion

from openstef_core import versioning
from openstef_core.versioning import get_runtime_openstef_version, is_openstef_compatible


@pytest.mark.parametrize(
    ("model_version", "runtime_version", "expected"),
    [
        pytest.param("4.4.0", "4.4.6", True, id="patch-difference-is-compatible"),
        pytest.param("4.3.6", "4.4.6", False, id="minor-difference-is-incompatible"),
        pytest.param("3.4.6", "4.4.6", False, id="major-difference-is-incompatible"),
        pytest.param(None, "4.4.6", False, id="missing-version-is-incompatible"),
        pytest.param("invalid", "4.4.6", False, id="invalid-version-is-incompatible"),
    ],
)
def test_is_openstef_compatible_matches_major_minor(
    model_version: str | None,
    runtime_version: str,
    expected: bool,
) -> None:
    # Act
    compatible = is_openstef_compatible(model_version=model_version, runtime_version=runtime_version)

    # Assert
    assert compatible is expected


def test_get_runtime_openstef_version_reads_core_distribution(monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange
    requested_distributions: list[str] = []

    def package_version(distribution_name: str) -> str:
        requested_distributions.append(distribution_name)
        return "4.4.6"

    monkeypatch.setattr(versioning, "package_version", package_version)

    # Act
    runtime_version = get_runtime_openstef_version()

    # Assert
    assert runtime_version == "4.4.6"
    assert requested_distributions == ["openstef-core"]


def test_is_openstef_compatible_infers_runtime_version(monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange
    monkeypatch.setattr(versioning, "package_version", lambda distribution_name: "4.4.6")

    # Act
    compatible = is_openstef_compatible(model_version="4.4.0")

    # Assert
    assert compatible


def test_is_openstef_compatible_raises_for_invalid_runtime_version() -> None:
    with pytest.raises(InvalidVersion):
        is_openstef_compatible(model_version="4.4.6", runtime_version="invalid")


def test_get_runtime_openstef_version_raises_when_core_distribution_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange
    def missing_distribution(distribution_name: str) -> str:
        raise PackageNotFoundError(distribution_name)

    monkeypatch.setattr(versioning, "package_version", missing_distribution)

    # Act / Assert
    with pytest.raises(PackageNotFoundError, match="openstef-core"):
        get_runtime_openstef_version()

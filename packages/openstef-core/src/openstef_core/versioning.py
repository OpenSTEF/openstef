# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""OpenSTEF package version provenance and compatibility."""

from importlib.metadata import version as package_version

from packaging.version import InvalidVersion, Version

OPENSTEF_VERSION_TAG = "openstef_version"


def get_runtime_openstef_version() -> str:
    """Return the installed OpenSTEF core distribution version.

    Raises:
        PackageNotFoundError: If the ``openstef-core`` distribution is not installed.
    """
    return package_version("openstef-core")


def is_openstef_compatible(*, model_version: str | None, runtime_version: str | None = None) -> bool:
    """Return whether model and runtime versions share their major and minor versions.

    Raises:
        InvalidVersion: If the runtime version is invalid.
        PackageNotFoundError: If the runtime version is inferred but ``openstef-core`` is not installed.
    """
    runtime = Version(runtime_version if runtime_version is not None else get_runtime_openstef_version())
    if model_version is None:
        return False

    try:
        model = Version(model_version)
    except InvalidVersion:
        return False

    return model.major == runtime.major and model.minor == runtime.minor


__all__ = ["OPENSTEF_VERSION_TAG", "get_runtime_openstef_version", "is_openstef_compatible"]

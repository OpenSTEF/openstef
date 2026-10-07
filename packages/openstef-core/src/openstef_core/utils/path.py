# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Utilities for safely constructing local filesystem paths."""

from pathlib import Path


def safe_path_join(base_path: Path, *path_components: str) -> Path:
    """Join path components and reject paths that escape the base directory."""
    resolved_base = base_path.resolve()
    joined_path = base_path.joinpath(*path_components)
    resolved_path = joined_path.resolve()

    try:
        resolved_path.relative_to(resolved_base)
    except ValueError as error:
        msg = f"Path escapes base directory: {resolved_path}"
        raise ValueError(msg) from error

    return joined_path

# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

from pathlib import Path

import pytest

from openstef_core.utils.path import safe_path_join


def test_safe_path_join_returns_path_under_base(tmp_path: Path) -> None:
    expected_path = tmp_path / "group" / "target"

    result = safe_path_join(tmp_path, "group", "target")

    assert result == expected_path


@pytest.mark.parametrize(
    "component",
    [
        pytest.param("../outside", id="parent-directory"),
        pytest.param("/outside", id="absolute-path"),
    ],
)
def test_safe_path_join_rejects_paths_outside_base(tmp_path: Path, component: str) -> None:
    with pytest.raises(ValueError, match="escapes base directory"):
        safe_path_join(tmp_path, component)


def test_safe_path_join_rejects_symlink_escape(tmp_path: Path) -> None:
    outside_path = tmp_path.parent / f"{tmp_path.name}-outside"
    outside_path.mkdir()
    (tmp_path / "link").symlink_to(outside_path, target_is_directory=True)

    try:
        with pytest.raises(ValueError, match="escapes base directory"):
            safe_path_join(tmp_path, "link", "file")
    finally:
        outside_path.rmdir()

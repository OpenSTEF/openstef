# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Tests for iterator utilities."""

from collections.abc import Callable
from typing import Any

import pytest

from openstef_core.utils.itertools import groupby, is_all_same, merge_iterators


def ascending_compare(a: int, b: int) -> int:
    """Compare function for ascending order."""
    return a - b


def descending_compare(a: int, b: int) -> int:
    """Compare function for descending order."""
    return b - a


@pytest.mark.parametrize(
    ("it1_items", "it2_items", "compare_func", "expected"),
    [
        pytest.param([1, 3, 5], [2, 4, 6], ascending_compare, [1, 2, 3, 4, 5, 6], id="sorted_numbers"),
        pytest.param([], [], ascending_compare, [], id="empty_iterators"),
        pytest.param([1, 3, 5], [], ascending_compare, [1, 3, 5], id="one_empty"),
        pytest.param([1, 2, 3], [2, 3, 4], ascending_compare, [1, 2, 2, 3, 3, 4], id="equal_elements"),
        pytest.param([5, 3, 1], [6, 4, 2], descending_compare, [6, 5, 4, 3, 2, 1], id="reverse_order"),
    ],
)
def test_merge_iterators(
    it1_items: list[int], it2_items: list[int], compare_func: Callable[[int, int], int], expected: list[int]
) -> None:
    # Arrange
    it1 = iter(it1_items)
    it2 = iter(it2_items)

    # Act
    result = list(merge_iterators(it1, it2, compare_func))

    # Assert
    assert result == expected


@pytest.mark.parametrize(
    ("items", "expected"),
    [
        pytest.param([1, 1, 1], True, id="all_same_numbers"),
        pytest.param([1, 2, 1], False, id="different_numbers"),
        pytest.param(["a", "a", "a"], True, id="all_same_strings"),
        pytest.param(["a", "b", "a"], False, id="different_strings"),
        pytest.param([1], True, id="single_item"),
        pytest.param(
            [], False, id="empty_list"
        ),  # Bug: empty set has length 0, function returns False but logically should be True
    ],
)
def test_is_all_same(items: list[Any], expected: bool) -> None:
    # Arrange / Act
    result: bool = is_all_same(items)

    # Assert
    assert result == expected


def test_is_all_same_with_none() -> None:
    # Arrange
    items = [None, None, None]

    # Act
    result: bool = is_all_same(items)

    # Assert
    assert result is True


@pytest.mark.parametrize(
    ("items", "expected"),
    [
        pytest.param(
            [("a", 1), ("b", 2), ("a", 3), ("c", 4), ("b", 5)],
            {"a": [1, 3], "b": [2, 5], "c": [4]},
            id="basic_grouping",
        ),
        pytest.param([], {}, id="empty_input"),
        pytest.param([("key", 1), ("key", 2), ("key", 3)], {"key": [1, 2, 3]}, id="single_group"),
        pytest.param([(1, "a"), (2, "b"), (1, "c")], {1: ["a", "c"], 2: ["b"]}, id="mixed_key_types"),
        pytest.param([("a", 1), ("a", 2), ("a", 3)], {"a": [1, 2, 3]}, id="preserves_order"),
    ],
)
def test_groupby(items: list[tuple[Any, Any]], expected: dict[Any, list[Any]]) -> None:
    # Arrange / Act
    result = groupby(items)

    # Assert
    assert result == expected

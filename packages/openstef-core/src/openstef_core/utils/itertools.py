# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Iterator utilities for data processing and stream merging.

Provides specialized iterator functions for merging sorted streams, grouping
operations, and validation checks commonly needed in time series data processing
and forecasting workflows.
"""

from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator


def merge_iterators[T](it1: Iterator[T], it2: Iterator[T], compare: Callable[[T, T], int]) -> Iterator[T]:
    """Yield items from two sorted iterators in order defined by compare(a, b).

    Args:
        it1: First sorted iterator
        it2: Second sorted iterator
        compare: Function that returns negative, zero, or positive value
                comparing two items (like cmp from Python 2)

    Yields:
        Items from both iterators in sorted order according to compare function

    Example:
        >>> def int_compare(a: int, b: int) -> int:
        ...     return a - b
        >>> list1 = [1, 3, 5]
        >>> list2 = [2, 4, 6]
        >>> list(merge_iterators(iter(list1), iter(list2), int_compare))
        [1, 2, 3, 4, 5, 6]

        >>> # Reverse order comparison
        >>> def reverse_compare(a: int, b: int) -> int:
        ...     return b - a
        >>> list3 = [5, 3, 1]  # sorted in descending order
        >>> list4 = [6, 4, 2]  # sorted in descending order
        >>> list(merge_iterators(iter(list3), iter(list4), reverse_compare))
        [6, 5, 4, 3, 2, 1]
    """
    a: T | None = next(it1, None)
    b: T | None = next(it2, None)

    # While both have items, pick the lesser (or equal) one
    while a is not None and b is not None:
        if compare(a, b) <= 0:
            yield a
            a = next(it1, None)
        else:
            yield b
            b = next(it2, None)

    # Drain the remainder of whichever iterator is left
    while a is not None:
        yield a
        a = next(it1, None)

    while b is not None:
        yield b
        b = next(it2, None)


def is_all_same[T](items: Iterable[T]) -> bool:
    """Check that all items have the same value.

    Note:
        This function requires that the items are hashable and comparable
        for equality. For unhashable types, consider converting to a list
        and comparing manually.

    Returns:
        True if all items in the iterable are equal, False otherwise.
        Returns False for empty iterables (no items to be equal).

    Example:
        >>> is_all_same([1, 1, 1, 1])
        True
        >>> is_all_same([1, 2, 1])
        False
        >>> is_all_same(['a', 'a', 'a'])
        True
        >>> is_all_same([])
        False
    """
    return len(set(items)) == 1


def groupby[K, V](iterable: Iterable[tuple[K, V]]) -> dict[K, list[V]]:
    """Group items by a key function.

    Returns:
        Dictionary mapping keys to lists of associated values.
    """
    grouped: defaultdict[K, list[V]] = defaultdict(list)
    for key, value in iterable:
        grouped[key].append(value)

    return grouped


__all__ = [
    "groupby",
    "is_all_same",
    "merge_iterators",
]

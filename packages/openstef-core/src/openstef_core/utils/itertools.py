# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator


def merge_iterators[T](it1: Iterator[T], it2: Iterator[T], compare: Callable[[T, T], int]) -> Iterator[T]:
    """Yield items from two sorted iterators in order defined by compare(a, b)."""
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
    """Check that all items have the same value."""
    return len(set(items)) == 1


def groupby[K, V](iterable: Iterable[tuple[K, V]]) -> dict[K, list[V]]:
    """Group items by a key function."""
    grouped: defaultdict[K, list[V]] = defaultdict(list)
    for key, value in iterable:
        grouped[key].append(value)

    return grouped


__all__ = [
    "groupby",
    "is_all_same",
    "merge_iterators",
]

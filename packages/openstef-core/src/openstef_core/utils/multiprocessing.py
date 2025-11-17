# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Multiprocessing utilities for parallel execution of tasks.

Provides simplified parallel processing functions with automatic platform-specific
optimizations for compute-intensive operations like model training and evaluation
across multiple forecasting scenarios.
"""

import multiprocessing
from collections.abc import Callable, Iterable
from typing import Literal


def run_parallel[T, R](
    process_fn: Callable[[T], R],
    items: Iterable[T],
    n_processes: int | None = None,
    mode: Literal["loky", "spawn", "fork"] = "loky",
) -> list[R]:
    """Execute a function in parallel across multiple processes.

    On macOS, explicitly uses fork context to avoid issues with the default
    spawn context that became the default in Python 3.8+. Fork context preserves
    the parent process memory, making it more efficient for sharing large objects
    like trained models or data structures.

    Args:
        process_fn: Function to apply to each item. Must be picklable for
                   multiprocessing. Lambda functions won't work - use def functions.
        items: Iterable of items to process.
        n_processes: Number of processes to use. If None or <= 1, runs sequentially.
                    Typically set to number of CPU cores or logical cores.
        mode: Multiprocessing start method. 'loky' is recommeneded for robust
                ml use-cases. 'fork' is more efficient on macOS, while 'spawn' is
                default on Windows/Linux. Xgboost seems to have bugs
                when used with 'fork'.

    Returns:
        List of results from applying process_fn to each item, in the same order
        as the input items.

    Example:
        >>> def square(x: int) -> int:
        ...     return x * x

        >>> # Sequential execution (n_processes <= 1) - always works
        >>> run_parallel(square, [1, 2, 3, 4], n_processes=1)
        [1, 4, 9, 16]

        >>> # For parallel execution, use module-level functions:
        >>> # run_parallel(math.sqrt, [1, 4, 9, 16], n_processes=2)
        >>> # [1.0, 2.0, 3.0, 4.0]

        >>> # Empty input handling
        >>> run_parallel(square, [], n_processes=1)
        []
    """
    if n_processes is None or n_processes <= 1:
        # If only one process is requested, run the function sequentially
        return [process_fn(item) for item in items]

    if mode == "loky":
        from joblib import Parallel, delayed  # pyright: ignore[reportUnknownVariableType] # noqa: PLC0415

        # Use joblib with loky backend for robust process management
        return Parallel(n_jobs=n_processes, backend="loky")(  # pyright: ignore[reportUnknownVariableType]
            delayed(process_fn)(item) for item in items
        )  # type: ignore

    # Auto-configure for macOS
    context = multiprocessing.get_context(method=mode)

    with context.Pool(processes=n_processes) as pool:
        return pool.map(process_fn, items)

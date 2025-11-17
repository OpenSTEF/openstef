# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Multiprocessing utilities for parallel execution of tasks.

Provides simplified parallel processing functions with automatic platform-specific
optimizations for compute-intensive operations like model training and evaluation
across multiple forecasting scenarios.
"""

import multiprocessing
import sys
from collections.abc import Callable, Iterable


def run_parallel[T, R](process_fn: Callable[[T], R], items: Iterable[T], n_processes: int | None = None) -> list[R]:
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

    Note:
        macOS Implementation Details:
        - Uses fork context instead of spawn to avoid serialization overhead
        - Fork preserves parent memory space, including imported modules and variables
        - More efficient for ML models and large data structures
        - On other platforms, uses the default context (usually spawn on Windows/Linux)
    """
    if n_processes is None or n_processes <= 1:
        # If only one process is requested, run the function sequentially
        return [process_fn(item) for item in items]

    # Auto-configure for macOS
    if sys.platform == "darwin":
        context = multiprocessing.get_context("fork")
    else:
        context = multiprocessing.get_context()

    with context.Pool(processes=n_processes) as pool:
        return pool.map(process_fn, items)

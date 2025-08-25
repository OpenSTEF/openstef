# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

import multiprocessing
import sys
from collections.abc import Callable, Iterable


def run_parallel[T, R](process_fn: Callable[[T], R], items: Iterable[T], n_processes: int | None = None) -> list[R]:
    if n_processes is None or n_processes <= 1:
        # If only one process is requested, run the function sequentially
        return [process_fn(item) for item in items]

    # Auto-configure for macOS
    if sys.platform == "darwin":
        context = multiprocessing.get_context("fork")
    else:
        context = multiprocessing.get_context()

    with context.Pool(processes=n_processes) as pool:
        results = pool.map(process_fn, items)
    return results

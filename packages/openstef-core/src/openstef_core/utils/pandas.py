# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

from datetime import datetime

import pandas as pd


def sorted_range_slice_idxs(data: pd.Series, start: datetime | None, end: datetime | None) -> tuple[int, int]:
    """Get sorted slice indices for a datetime range."""
    start_idx = data.searchsorted(start, side="left") if start else 0
    end_idx = data.searchsorted(end, side="left") if end else len(data)
    return start_idx, end_idx
